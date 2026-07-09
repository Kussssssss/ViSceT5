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

        if mode in ["full", "all"]:
            total_loss = mlm_loss + pollute_loss + contrastive_loss

        elif mode in ["gen_all", "gen"]:
            # Encoder-head MLM is DROPPED (the discarded-at-finetune BERT-ism from the
            # TWA port). Masked-prediction is now done GENERATIVELY by the decoder via
            # grounded-cloze (T5 span-infill) — which trains the decoder finetune uses.
            # Objective = ITM + TWC (encoder aux) + cloze (decoder). mlm_loss is 0 here
            # (cmb_text_mask_label all -1) and left out of the total by design.
            total_loss = pollute_loss + contrastive_loss
            if gen_loss is not None:
                total_loss = total_loss + gen_loss
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
    Retrieval-based TWC accuracy, aligned with the softmax (InfoNCE) objective.

    For each REAL original OCR token (row, PAD excluded), we take the highest-
    similarity augmented token (argmax over valid columns) and count it correct
    if that column is a true positive (label > 0.5). This directly measures
    whether the model ranks a token's own augmented counterpart above all the
    in-batch negatives — the actual goal of TWC. Returns:
        (retrieval_acc, top1_pos_rate, top1_neg_rate)
    where the latter two are the row-wise top-1 hit/miss rates kept for the
    8-element diagnostic vector (twc_pos_recall / twc_neg_recall slots).
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
        pos_rate = retrieval_acc                    # fraction of rows whose top-1 is positive
        neg_rate = 1.0 - retrieval_acc              # fraction whose top-1 is a negative/wrong
        return retrieval_acc, pos_rate, neg_rate


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

        # TRẢ VỀ 1 TENSOR CHỨA 14 GIÁ TRỊ ĐỂ TRAINER THU THẬP
        # [0] total_acc  [1] acc_slot1 (ITM-only in cloze; (MLM+ITM)/2 legacy)  [2] twc_acc
        # [3] loss_mlm (0 in cloze) [4] loss_itm [5] loss_twc  [6] twc_pos_recall
        # [7] twc_neg_recall  [8] loss_gen (=decoder MLM loss)  [9] mlm_acc
        # [10] g_correct [11] g_total [12] r_correct [13] r_total (grounded/random counts)
        device = model_output["textcls_scores"].device
        return torch.tensor(
            [total_acc, acc_slot1, twc_acc,
             loss_mlm, loss_itm, loss_twc,
             twc_pos_recall, twc_neg_recall, loss_gen, cloze_acc,
             g_correct, g_total, r_correct, r_total],
            device=device
        )