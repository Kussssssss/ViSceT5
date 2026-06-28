# @title
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

def _twc_infonce(logits, labels, valid):
    """
    Soft-label InfoNCE for one direction of TWC (TWA Eq. 4–6).

    logits : [M, M] = logit_scale * <O_i, R_j>  (cosine, scaled by temperature)
    labels : [M, M] soft labels in {-1 (ignore), 0 (neg), 0.9 (semi), 1.0 (pos)}
    valid  : [M] bool — real (non-PAD) tokens; PAD rows AND cols are excluded.

    For each valid row i we softmax over the valid columns (the candidate
    augmented tokens in the batch) and take cross-entropy against the row's
    soft-label distribution. Negatives (other tokens / other images) live in the
    softmax denominator, so the positive must out-rank them — this is what keeps
    the task from collapsing to "predict everything negative" (the failure mode
    of independent per-cell sigmoid BCE under heavy class imbalance).
    """
    neg_inf = -1e9  # safe large-negative mask (far from fp32 overflow)
    col_mask = valid.unsqueeze(0)                        # [1, M] valid columns
    masked_logits = logits.masked_fill(~col_mask, neg_inf)
    logp = torch.log_softmax(masked_logits, dim=1)       # over candidate columns

    tgt = labels.clamp(min=0.0) * col_mask.float()       # -1 -> 0, drop invalid cols
    denom = tgt.sum(dim=1, keepdim=True)                 # positive mass per row
    has_pos = (denom.squeeze(1) > 0) & valid             # valid rows that own a positive
    tgt = tgt / denom.clamp_min(1e-9)                    # normalize to a distribution

    row_ce = -(tgt * logp).sum(dim=1)                    # [M] cross-entropy per row
    row_ce = row_ce[has_pos]
    if row_ce.numel() == 0:
        return logits.sum() * 0.0                        # keep in graph, no signal
    return row_ce.mean()


class ViT5PretrainLoss(nn.Module):
    def __init__(self, pretrain_ablation_mode="full"):
        super().__init__()
        self.pretrain_ablation_mode = str(pretrain_ablation_mode).lower().strip()

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

            # TWA Eq. 4–6: softmax (temperature) contrastive with soft labels,
            # over REAL tokens only. A token is PAD/ignored iff its diagonal == -1
            # (create_batch_labels has propagated -1 across its row/col).
            diag = torch.diagonal(o2r_block)
            valid = diag != -1

            n_valid = int(valid.sum().item())
            if n_valid >= 2:
                o2r_labels = o2r_block.float()
                r2o_labels = r2o_block.float()
                loss_i = _twc_infonce(logits_per_image.float(), o2r_labels, valid)  # token->word
                loss_t = _twc_infonce(logits_per_text.float(),  r2o_labels, valid)  # word->token
                contrastive_loss = (loss_i + loss_t) / 2
            else:
                # Degenerate batch with <2 real OCR tokens — TWC cannot form a
                # contrastive pair. Never silent: warn loudly so it can't hide.
                print(f"⚠️ [TWC] only {n_valid} real OCR token(s) in batch — TWC contributes 0 "
                      f"this step (check OCR extraction / augmentation if this recurs).")
                contrastive_loss = logits_per_image.sum() * 0.0

        mode = self.pretrain_ablation_mode

        if mode in ["full", "all"]:
            total_loss = mlm_loss + pollute_loss + contrastive_loss

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

        # 3. Tính Total Acc tùy mode
        if self.mode == "only_twc_ocr_aug":
            total_acc = twc_acc
        elif self.mode in ["no_twc_ocr_aug", "only_itm_mlm"]:
            total_acc = (mlm_acc + itm_acc) / 2.0
        else:
            total_acc = (mlm_acc + itm_acc + twc_acc) / 3.0

        # TRẢ VỀ 1 TENSOR CHỨA 8 GIÁ TRỊ ĐỂ TRAINER THU THẬP
        # [0] total_acc  [1] mlm_itm_acc  [2] twc_acc  [3] loss_mlm
        # [4] loss_itm   [5] loss_twc     [6] twc_pos_recall  [7] twc_neg_recall
        device = model_output["textcls_scores"].device
        return torch.tensor(
            [total_acc, (mlm_acc + itm_acc)/2.0, twc_acc,
             loss_mlm, loss_itm, loss_twc,
             twc_pos_recall, twc_neg_recall],
            device=device
        )