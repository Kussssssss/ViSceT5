# @title
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# @title
def create_batch_labels(batch_labels):
    if isinstance(batch_labels, torch.Tensor):
        if batch_labels.dim() == 2:
            result = batch_labels.clone()
        elif batch_labels.dim() == 3:
            B, L, _ = batch_labels.shape
            blocks = [batch_labels[b] for b in range(B)]
            result = torch.block_diag(*blocks)          # (B*L, B*L)
        else:
            raise ValueError(f"Unsupported batch_labels dim: {batch_labels.dim()}")
    else:
        blocks = batch_labels
        result = torch.block_diag(*blocks)          # (B*L, B*L)

    diag = torch.diagonal(result)               # (B*L,)
    ignore_mask = (diag == -1)
    if ignore_mask.any():
        result[ignore_mask, :] = -1.0           # row i
        result[:, ignore_mask] = -1.0           # col i
    return result

class ViT5PretrainLoss(nn.Module):
    def __init__(self, pretrain_ablation_mode="full"):
        super().__init__()
        self.pretrain_ablation_mode = str(pretrain_ablation_mode).lower().strip()
        # pos_weight is NOT in TWA (TWA = plain BCE). Toggle via env:
        #   TWC_POS_WEIGHT_CAP unset / 0  -> plain BCE (TWA-faithful)
        #   TWC_POS_WEIGHT_CAP = e.g. 1000 -> dynamic pos_weight = n_neg/n_pos, clamped [1, cap]
        try:
            self.twc_pos_weight_cap = float(os.environ.get("TWC_POS_WEIGHT_CAP", "0"))
        except ValueError:
            self.twc_pos_weight_cap = 0.0
        # ITM weight in the gen_all total. ITM (pollute) stays stuck at chance (loss ≈
        # ln2) — the image↔OCR mismatch it must detect needs fine visual grounding that
        # the FROZEN coarse CLIP (+ no_grad/nan_to_num in pretrain) can't provide, and it
        # reads only the decoder start-token hidden. Its head is also discarded at
        # finetune. So DEFAULT it OFF (0) as dead weight; set ITM_WEIGHT=1 to restore.
        try:
            self.itm_weight = float(os.environ.get("ITM_WEIGHT", "0"))
        except ValueError:
            self.itm_weight = 0.0
        # ITC (Image-Text Contrastive, image↔question) weight. Opt-in (default 0);
        # set ITC_WEIGHT>0 to enable align-before-fuse. Pairs with vision unfreeze.
        try:
            self.itc_weight = float(os.environ.get("ITC_WEIGHT", "0"))
        except ValueError:
            self.itc_weight = 0.0
        # ITC memory queue (MoCo-style negatives, NO momentum encoder): keep the last
        # K L2-normalized image/text vectors as extra negatives. With per_device
        # batch 4, in-batch ITC has only 3 negatives → trivially separable (acc≈0.94
        # at epoch 1) → weak alignment signal; a queue makes it a real task. Slightly
        # stale negatives are the standard, acceptable trade-off at low LR.
        # Env ITC_QUEUE=K (0/unset = off = pure in-batch, current behavior).
        try:
            self.itc_queue_size = int(float(os.environ.get("ITC_QUEUE", "0") or 0))
        except ValueError:
            self.itc_queue_size = 0
        self._itc_img_q = None   # (≤K, D) detached, CPU
        self._itc_txt_q = None
        # FALSE-NEGATIVE mask cho câu hỏi TEMPLATE (đặc thù ViTextVQA: rất nhiều ảnh
        # khác nhau mang câu hỏi Y HỆT "cửa hàng này tên gì ?"). Candidate j có text
        # TRÙNG text của query i KHÔNG phải negative thật — nếu vẫn phạt, ITC học
        # phân biệt tuỳ tiện giữa các chuỗi identical → đặc trưng nhiễu.
        # CÁCH LÀM: so TRÙNG CHÍNH XÁC bằng hash token-ids của câu hỏi (bất biến với
        # training). KHÔNG dùng cosine trên vector text đang học — mock đã chứng minh
        # model GIAN LẬN được: co cụm mọi vector text → mọi negative cos>tau → bị mask
        # sạch → loss_itc=0/acc=1 miễn phí (degenerate). Hash thì không thể cheat.
        # Queue lưu (img, txt, key) THEO CẶP nên 1 mask key↔key dùng cho cả 2 chiều;
        # positive của chính nó luôn giữ. Env ITC_DUP_TAU (>0 = bật; 0/unset = off).
        try:
            self.itc_dup_tau = float(os.environ.get("ITC_DUP_TAU", "0") or 0)
        except ValueError:
            self.itc_dup_tau = 0.0
        # v4: nguồn text của ITC — 'question' (v3) | 'ocr' (image↔chuỗi-OCR; xem model).
        # Quyết định text-key dùng cho dup-mask: hash câu hỏi vs hash OCR-word-ids.
        self.itc_text_source = os.environ.get("ITC_TEXT_SOURCE", "question").strip().lower()
        self._itc_key_q = None    # (≤K,) int64 hash NỘI DUNG text-side, thẳng hàng 2 queue
        # v4: same-image mask — 1 ảnh ~3 câu hỏi thành 3 sample; khi anchor là (ảnh X,
        # text1) mà sample anh em của CÙNG X nằm trong queue thì nó KHÔNG phải negative
        # (vẫn thuộc về X — ~22% anchor dính với queue 4096 trên 35k sample). Quan hệ
        # của sample anh em vẫn được HỌC khi chính nó làm anchor; mask chỉ gỡ gradient
        # mâu thuẫn. Key = hash(image_path từ collator) — training-invariant, không cheat.
        self._itc_ikey_q = None   # (≤K,) int64 hash ảnh, thẳng hàng 2 queue

    def forward(self, sample_list, model_output):
        for k, v in model_output.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                print(f"🚨 NAN DETECTED IN MODEL OUTPUT: {k}")

        if "textcls_scores" not in model_output:
            device = sample_list.get("tag_pollute", torch.tensor(0)).device
            return torch.tensor(0.0, device=device, requires_grad=True)

        tcls = model_output["textcls_scores"].float()  # (B, L, V)
        targets = sample_list["cmb_text_mask_label"].to(tcls.device).long()

        loss_mask = (targets != -1).float()

        pollute = sample_list["tag_pollute"].to(tcls.device).float()
        if pollute.ndim > 1:
            pollute = pollute.squeeze(-1)

        keep = (1.0 - pollute).unsqueeze(1)
        final_mask = loss_mask * keep

        scores = tcls.permute(0, 2, 1)

        mlm_losses = F.cross_entropy(
            scores,
            targets,
            reduction="none",
            ignore_index=-1,
        )

        masked_losses = mlm_losses * final_mask
        denom = final_mask.sum().clamp_min(1.0)
        mlm_loss = masked_losses.sum() / denom

        p_scores = model_output["pollutecls_scores"].float()
        p_targets = sample_list["tag_pollute"].to(p_scores.device).float()

        if p_scores.ndim > 1:
            p_scores = p_scores.squeeze(-1)
        if p_targets.ndim > 1:
            p_targets = p_targets.squeeze(-1)

        pollute_loss = F.binary_cross_entropy_with_logits(
            p_scores,
            p_targets,
            reduction="mean",
        )

        # Giữ tensor này nằm trong graph để tránh backward error
        contrastive_loss = mlm_loss * 0.0

        logits_per_image = model_output.get("contrastive_scores", None)
        o2r_block = model_output.get("o2r_block", None)
        r2o_block = model_output.get("r2o_block", None)

        if logits_per_image is not None and o2r_block is not None:
            logits_per_text = logits_per_image.t()

            o2r_block = o2r_block.to(logits_per_image.device)
            o2r_block = create_batch_labels(o2r_block)

            if r2o_block is None:
                r2o_block = o2r_block.transpose(0, 1)
            else:
                r2o_block = r2o_block.to(logits_per_image.device)
                r2o_block = create_batch_labels(r2o_block)

            # TWA released code: sigmoid BCE on the valid (non-PAD) cells of the
            # block-diagonal label matrix. PAD/ignored cells are excluded via
            # mask != -1 (our collator sets PAD diagonal = -1; create_batch_labels
            # then propagates -1 across PAD rows/cols). Same-text in-image pairs are
            # positive/semi-positive; everything else (cross-text, cross-image) = 0.
            mask = o2r_block != -1
            if mask.any():
                o2r_labels = o2r_block.float()
                r2o_labels = r2o_block.float()
                o2r_m, r2o_m = o2r_labels[mask], r2o_labels[mask]

                # TWA = plain BCE (no pos_weight). pos_weight is an optional imbalance
                # fix, toggled via env TWC_POS_WEIGHT_CAP (0/unset => plain BCE).
                cap = self.twc_pos_weight_cap
                if cap and cap > 1.0:
                    def _pos_weight(lbl):
                        n_pos = (lbl > 0.5).sum().clamp_min(1.0)
                        n_neg = (lbl == 0.0).sum().clamp_min(1.0)
                        return (n_neg / n_pos).clamp(1.0, cap)
                    pw_i, pw_t = _pos_weight(o2r_m), _pos_weight(r2o_m)
                else:
                    pw_i = pw_t = None  # plain BCE, TWA-faithful

                loss_i = F.binary_cross_entropy_with_logits(
                    logits_per_image[mask].float(), o2r_m, pos_weight=pw_i, reduction="mean")
                loss_t = F.binary_cross_entropy_with_logits(
                    logits_per_text[mask].float(), r2o_m, pos_weight=pw_t, reduction="mean")
                contrastive_loss = (loss_i + loss_t) / 2
            else:
                # No valid (non-PAD) contrastive cells — never silent; warn loudly.
                print("⚠️ [TWC] no valid (non-PAD) contrastive cells this batch — TWC contributes 0.")
                contrastive_loss = logits_per_image.sum() * 0.0

        mode = self.pretrain_ablation_mode

        # Grounded-cloze decoder loss. Stamped by the trainer in 'gen'/'gen_all'
        # modes. This is what trains the seq2seq DECODER that finetune actually uses
        # (MLM/ITM/TWC only shape the encoder): the decoder regenerates the CLEAN
        # question words that appear in the OCR, forcing it to read the OCR feature
        # branch. See pretrain_decoder_plan.md. (key name 'gen_loss' kept for compat.)
        gen_loss = model_output.get("gen_loss", None)

        # ITC (Image-Text Contrastive, image↔question): in-batch symmetric InfoNCE.
        # Aligns image and question in a shared space (ALBEF align-before-fuse). Opt-in.
        itc_loss = None
        itc_acc = None
        _iv, _tv = model_output.get("itc_img_vec"), model_output.get("itc_txt_vec")
        if self.itc_weight > 0 and _iv is not None and _tv is not None and _iv.size(0) > 1:
            _sc = model_output.get("itc_logit_scale")
            _sc = _sc.exp().clamp(max=100.0) if torch.is_tensor(_sc) else 14.3
            _l_i2t = _sc * (_iv @ _tv.t())             # (B, B): image i ↔ question j
            _l_t2i = _sc * (_tv @ _iv.t())
            # Queue negatives on TRAIN forwards only (grad on): eval acc_itc stays
            # pure in-batch → comparable across steps/runs; queue never sees eval vecs.
            # FALSE-NEGATIVE keys — training-invariant hashes → un-cheatable (see
            # __init__). Two keys, masked as a UNION:
            #   text-key : hash NỘI DUNG text-side (question ids, hoặc clean-OCR word
            #              ids khi ITC_TEXT_SOURCE=ocr) — bản trùng nguyên văn.
            #   image-key: hash image_path — sample anh em CÙNG ẢNH (1 ảnh ~3 câu hỏi).
            _keys = _ikeys = None
            if self.itc_dup_tau > 0:
                _B0 = _iv.size(0)
                if self.itc_text_source == "ocr":
                    _wid = sample_list.get("twa_word_ids")
                    _wmap = sample_list.get("ocr_to_word_map")
                    _char = sample_list.get("twa_ocr_char")
                    if _wid is not None and _wmap is not None and _wid.size(0) == _B0:
                        _half = (_char.size(1) // 2
                                 if (_char is not None and sample_list.get("o2r_labels") is not None)
                                 else (int(_wmap.max().item()) + 1 if _wmap.numel() else 0))
                        _wid_c = _wid.detach().cpu(); _wmap_c = _wmap.detach().cpu()
                        _keys = torch.tensor(
                            [hash(tuple(int(t) for t, m in zip(_wid_c[b], _wmap_c[b])
                                        if 0 <= int(m) < _half and int(t) != 0))
                             for b in range(_B0)], dtype=torch.long)
                else:
                    _q_ids = sample_list.get("mlm_input_ids")
                    if _q_ids is not None and _q_ids.size(0) == _B0:
                        _keys = torch.tensor(
                            [hash(tuple(int(t) for t in row[row != 0]))
                             for row in _q_ids.detach().cpu()], dtype=torch.long)
                _oinfo = sample_list.get("ocr_info")
                if isinstance(_oinfo, (list, tuple)) and len(_oinfo) == _B0:
                    _paths = [d.get("itc_image_path") if isinstance(d, dict) else None
                              for d in _oinfo]
                    if all(p is not None for p in _paths):
                        _ikeys = torch.tensor([hash(str(p)) for p in _paths], dtype=torch.long)
            _use_q = (self.itc_queue_size > 0 and _iv.requires_grad
                      and self._itc_txt_q is not None and self._itc_txt_q.size(0) > 0)
            if _use_q:
                _qi = self._itc_img_q.to(device=_iv.device, dtype=_iv.dtype)
                _qt = self._itc_txt_q.to(device=_tv.device, dtype=_tv.dtype)
                _l_i2t = torch.cat([_l_i2t, _sc * (_iv @ _qt.t())], dim=1)  # B×(B+K)
                _l_t2i = torch.cat([_l_t2i, _sc * (_tv @ _qi.t())], dim=1)
            # Union dup-mask: candidate trùng NỘI DUNG text HOẶC trùng ẢNH với query là
            # false negative ở CẢ 2 chiều → loại khỏi softmax; positive của chính nó giữ.
            _Bq = _tv.size(0)
            _ncand = _l_i2t.size(1)

            def _mk_dup(bk, qk):
                # bk: (B,) key của batch | qk: key trong queue (thẳng hàng vector queue)
                if bk is None:
                    return None
                ck = bk
                if _ncand > _Bq:
                    if qk is None or qk.size(0) != _ncand - _Bq:
                        return None  # queue lệch hàng → bỏ mask này (an toàn)
                    ck = torch.cat([bk, qk], dim=0)
                d = bk.unsqueeze(1).eq(ck.unsqueeze(0))
                d &= bk.unsqueeze(1) != 0         # key 0 = unknown → không bao giờ mask
                d[:, :_Bq].fill_diagonal_(False)  # giữ positive của chính nó
                return d

            _dup_t = _mk_dup(_keys, self._itc_key_q if _use_q else None)
            _dup_i = _mk_dup(_ikeys, self._itc_ikey_q if _use_q else None)
            _dup = _dup_t if _dup_i is None else (_dup_i if _dup_t is None else (_dup_t | _dup_i))
            if _dup is not None:
                _dup = _dup.to(_l_i2t.device)
                _l_i2t = _l_i2t.masked_fill(_dup, float("-inf"))
                _l_t2i = _l_t2i.masked_fill(_dup, float("-inf"))
            _tgt = torch.arange(_l_i2t.size(0), device=_l_i2t.device)
            itc_loss = 0.5 * (F.cross_entropy(_l_i2t, _tgt) + F.cross_entropy(_l_t2i, _tgt))
            # ITC retrieval accuracy: top-1 over B(+K) candidates, both directions.
            with torch.no_grad():
                _i2t = (_l_i2t.argmax(dim=1) == _tgt).float().mean()
                _t2i = (_l_t2i.argmax(dim=1) == _tgt).float().mean()
                itc_acc = 0.5 * (_i2t + _t2i)
                if self.itc_queue_size > 0 and _iv.requires_grad:
                    _di = _iv.detach().float().cpu()
                    _dt = _tv.detach().float().cpu()
                    self._itc_img_q = _di if self._itc_img_q is None else \
                        torch.cat([self._itc_img_q, _di], 0)[-self.itc_queue_size:]
                    self._itc_txt_q = _dt if self._itc_txt_q is None else \
                        torch.cat([self._itc_txt_q, _dt], 0)[-self.itc_queue_size:]
                    # keys enqueued in lockstep with the vector queues (fallback key 0 =
                    # "unknown"; _mk_dup never masks on key 0 so alignment stays safe)
                    _dk = _keys if _keys is not None else torch.zeros(_di.size(0), dtype=torch.long)
                    self._itc_key_q = _dk if self._itc_key_q is None else \
                        torch.cat([self._itc_key_q, _dk], 0)[-self.itc_queue_size:]
                    _dik = _ikeys if _ikeys is not None else torch.zeros(_di.size(0), dtype=torch.long)
                    self._itc_ikey_q = _dik if self._itc_ikey_q is None else \
                        torch.cat([self._itc_ikey_q, _dik], 0)[-self.itc_queue_size:]

        if mode in ["full", "all"]:
            total_loss = mlm_loss + pollute_loss + contrastive_loss

        elif mode in ["gen_all", "gen"]:
            # Encoder-head MLM is DROPPED (the discarded-at-finetune BERT-ism from the
            # TWA port). Masked-prediction is now done GENERATIVELY by the decoder via
            # grounded-cloze (T5 span-infill) — which trains the decoder finetune uses.
            # Objective = ITM + TWC (encoder aux) + cloze (decoder). mlm_loss is 0 here
            # (cmb_text_mask_label all -1) and left out of the total by design.
            total_loss = self.itm_weight * pollute_loss + contrastive_loss
            if gen_loss is not None:
                total_loss = total_loss + gen_loss
            if itc_loss is not None:
                total_loss = total_loss + self.itc_weight * itc_loss
            # else: batch produced no cloze span (rare with random-span fallback) →
            # train encoder aux (ITM+TWC) only; decoder skips this batch.

        elif mode in ["only_twc_ocr_aug", "no_mlm_itm", "w/o_mlm_itm", "without_mlm_itm"]:
            if logits_per_image is None or o2r_block is None:
                raise RuntimeError(
                    "only_twc_ocr_aug requires valid TWC outputs. "
                    "Please set use_ocr_aug=True and model.config.use_twc=True."
                )

            if not contrastive_loss.requires_grad:
                raise RuntimeError(
                    "contrastive_loss has no grad. Check o2r_labels/r2o_labels and TWC branch."
                )

            total_loss = contrastive_loss + (0.0 * mlm_loss) + (0.0 * pollute_loss)

        elif mode in ["no_twc_ocr_aug", "no_twc", "w/o_twc", "without_twc"]:
            total_loss = mlm_loss + pollute_loss

        else:
            raise ValueError(f"Unknown pretrain_ablation_mode: {mode}")

        model_output["loss_mlm"] = mlm_loss.detach()
        model_output["loss_itm"] = pollute_loss.detach()
        model_output["loss_twc"] = contrastive_loss.detach()
        model_output["loss_gen"] = (
            gen_loss.detach() if gen_loss is not None else mlm_loss.detach() * 0.0
        )
        model_output["loss_itc"] = (
            itc_loss.detach() if itc_loss is not None else mlm_loss.detach() * 0.0
        )
        model_output["acc_itc"] = (
            itc_acc.detach() if itc_acc is not None else mlm_loss.detach() * 0.0
        )
        return total_loss

class BaseMetric:
    def __init__(self, name):
        self.name = name
    def calculate(self, sample_list, model_output):
        raise NotImplementedError
    def __call__(self, sample_list, model_output):
        return self.calculate(sample_list, model_output)


class PreTrainContraAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("pollute_acc")

    def calculate(self, sample_list, model_output):
        if "pollutecls_scores" not in model_output:
            return 0.0
        scores = model_output["pollutecls_scores"].detach()
        targets = sample_list["tag_pollute"].to(scores.device).float().detach()

        if scores.ndim > 1: scores = scores.squeeze(-1)
        if targets.ndim > 1: targets = targets.squeeze(-1)

        preds = (torch.sigmoid(scores) > 0.5).float()
        correct = (preds == (targets > 0.5).float()).float()
        return correct.mean().item()


class PreTrainMLMAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("mlm_acc")

    def calculate(self, sample_list, model_output):
        if "textcls_scores" not in model_output:
            return 0.0
        logits = model_output["textcls_scores"].detach()
        targets = sample_list["cmb_text_mask_label"].to(logits.device).detach()

        B, L, V = logits.shape
        scores_flat = logits.reshape(B * L, V)
        targets_flat = targets.reshape(B * L)
        base_mask = targets_flat != -1

        pollute = sample_list["tag_pollute"].to(logits.device).float().detach()
        pollute_expanded = pollute.view(B, 1).expand(B, L).reshape(B * L)
        keep_mask = (1.0 - pollute_expanded) > 0.5
        valid_indices = base_mask & keep_mask

        if valid_indices.float().sum() == 0:
            return 0.0

        preds = scores_flat[valid_indices].argmax(dim=1)
        labels = targets_flat[valid_indices]
        return (preds == labels).float().mean().item()

class PreTrainTWCAccuracy(BaseMetric):
    """
    Retrieval-based TWC accuracy, aligned with the ranking goal of TWC.

    acc: for each REAL original OCR token (row, PAD excluded), take the highest-
    similarity augmented token (argmax over valid columns); correct if that column
    is a true positive (label > 0.5) — does the model rank a token's own augmented
    counterpart above all in-batch negatives?

    pos/neg recall: TRUE cell-level recalls at the BCE decision threshold
    (sigmoid>0.5 ⇔ logit>0), over cells whose row AND col are real (non-PAD):
        pos_recall = P(logit > 0 | label > 0.5)   — positives detected
        neg_recall = P(logit < 0 | label == 0)    — negatives rejected
    (Trước đây 2 slot này bị stamp bằng (acc, 1-acc) — trùng thông tin và gây
    hiểu sai; nay là recall thật, khớp đúng tên trường twc_pos/neg_recall.)

    Returns (retrieval_acc, pos_recall, neg_recall).
    """
    def __init__(self):
        super().__init__("twc_acc")

    def calculate(self, sample_list, model_output):
        cs = model_output.get("contrastive_scores", None)
        o2r = model_output.get("o2r_block", None)
        if cs is None or o2r is None:
            return 0.0, 0.0, 0.0

        logits = cs.detach()
        o2r = o2r.detach()
        valid = torch.diagonal(o2r) != -1          # real (non-PAD) tokens
        if int(valid.sum().item()) < 1:
            return 0.0, 0.0, 0.0

        # Mask invalid (PAD) columns out of the retrieval argmax.
        masked = logits.masked_fill(~valid.unsqueeze(0), -1e9)

        rows = valid.nonzero(as_tuple=True)[0]      # valid row indices
        pred = masked[rows].argmax(dim=1)           # top-1 column per valid row
        hit = (o2r[rows, pred] > 0.5).float()       # is the top-1 a true positive?
        retrieval_acc = hit.mean().item()

        # True cell-level recalls at the BCE threshold. Only cells with BOTH ends
        # real: torch.block_diag zero-fills cross-sample cells for PAD rows/cols too,
        # so gate on valid row & col (mirrors create_batch_labels' -1 propagation).
        cell_valid = valid.unsqueeze(0) & valid.unsqueeze(1)
        pos_m = cell_valid & (o2r > 0.5)
        neg_m = cell_valid & (o2r == 0.0)
        pos_recall = (logits[pos_m] > 0).float().mean().item() if bool(pos_m.any()) else 0.0
        neg_recall = (logits[neg_m] < 0).float().mean().item() if bool(neg_m.any()) else 0.0
        return retrieval_acc, pos_recall, neg_recall


# HÀM METRIC TỔNG - TỰ ĐỘNG ĐIỀU HƯỚNG THEO ABLATION MODE
class GlobalPretrainAccuracy(BaseMetric):
    def __init__(self, mode="all"):
        super().__init__("global_acc")
        self.mode = mode
        self.mlm_fn = PreTrainMLMAccuracy()
        self.itm_fn = PreTrainContraAccuracy()
        self.twc_fn = PreTrainTWCAccuracy()

    def calculate(self, sample_list, model_output):
        # 1. Lấy Acc
        mlm_acc = self.mlm_fn.calculate(sample_list, model_output)
        itm_acc = self.itm_fn.calculate(sample_list, model_output)
        twc_result = self.twc_fn.calculate(sample_list, model_output)
        if isinstance(twc_result, tuple):
            twc_acc, twc_pos_recall, twc_neg_recall = twc_result
        else:
            twc_acc, twc_pos_recall, twc_neg_recall = twc_result, 0.0, 0.0

        # 2. Lấy Loss (đã nhét vào từ bước 1)
        loss_mlm = model_output.get("loss_mlm", torch.tensor(0.0)).item()
        loss_itm = model_output.get("loss_itm", torch.tensor(0.0)).item()
        loss_twc = model_output.get("loss_twc", torch.tensor(0.0)).item()
        _lg = model_output.get("loss_gen", torch.tensor(0.0))
        loss_gen = _lg.item() if torch.is_tensor(_lg) else float(_lg or 0.0)
        # MLM (decoder span-infill) token accuracy — the decoder-side analog of the old
        # encoder mlm_acc.
        _ca = model_output.get("gen_acc", None)
        cloze_acc = _ca.item() if torch.is_tensor(_ca) else (float(_ca) if _ca is not None else 0.0)
        # Grounded/random split as token COUNTS (aggregator sums → exact token-weighted
        # acc, no NaN). grounded = masked because in OCR; random = LM-prior word.
        def _f(k):
            v = model_output.get(k, None)
            return v.item() if torch.is_tensor(v) else (float(v) if v is not None else 0.0)
        g_correct, g_total = _f("gen_g_correct"), _f("gen_g_total")
        r_correct, r_total = _f("gen_r_correct"), _f("gen_r_total")
        loss_itc = _f("loss_itc")
        acc_itc = _f("acc_itc")

        # 3. Tính Total Acc tùy mode
        if self.mode in ("gen", "gen_all"):
            # Cloze modes: encoder-head MLM is DROPPED (mlm_acc/loss_mlm ≡ 0 by design).
            # The masked-prediction accuracy is now CLOZE (decoder). Report ITM alone in
            # acc-slot; total = mean(ITM, TWC, CLOZE).
            acc_slot1 = itm_acc
            total_acc = (itm_acc + twc_acc + cloze_acc) / 3.0
        elif self.mode == "only_twc_ocr_aug":
            acc_slot1 = (mlm_acc + itm_acc) / 2.0
            total_acc = twc_acc
        elif self.mode in ["no_twc_ocr_aug", "only_itm_mlm"]:
            acc_slot1 = (mlm_acc + itm_acc) / 2.0
            total_acc = (mlm_acc + itm_acc) / 2.0
        else:
            acc_slot1 = (mlm_acc + itm_acc) / 2.0
            total_acc = (mlm_acc + itm_acc + twc_acc) / 3.0

        # TRẢ VỀ 1 TENSOR CHỨA 16 GIÁ TRỊ ĐỂ TRAINER THU THẬP
        # [0] total_acc  [1] acc_slot1 (ITM-only in cloze; (MLM+ITM)/2 legacy)  [2] twc_acc
        # [3] loss_mlm (0 in cloze) [4] loss_itm [5] loss_twc  [6] twc_pos_recall
        # [7] twc_neg_recall  [8] loss_gen (=decoder MLM loss)  [9] mlm_acc
        # [10] g_correct [11] g_total [12] r_correct [13] r_total (grounded/random counts)
        # [14] loss_itc (image↔question contrastive; 0 when ITC off)  [15] acc_itc (retrieval)
        device = model_output["textcls_scores"].device
        return torch.tensor(
            [total_acc, acc_slot1, twc_acc,
             loss_mlm, loss_itm, loss_twc,
             twc_pos_recall, twc_neg_recall, loss_gen, cloze_acc,
             g_correct, g_total, r_correct, r_total, loss_itc, acc_itc],
            device=device
        )