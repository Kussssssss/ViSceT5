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

            mask = o2r_block != -1

            if mask.any():
                o2r_labels = o2r_block.float()
                r2o_labels = r2o_block.float()

                # ── FIX: Dynamic pos_weight để cân bằng class imbalance ──────────────
                # Label matrix có ~99.9% negatives (label=0.0). Nếu không có pos_weight,
                # gradient hoàn toàn bị dominated bởi negatives khiến loss tầm thường.
                with torch.no_grad():
                    valid_o2r = o2r_labels[mask]
                    n_pos_o2r = (valid_o2r > 0.5).float().sum().clamp_min(1.0)
                    n_neg_o2r = (valid_o2r == 0.0).float().sum().clamp_min(1.0)
                    # Cap at 20: balanced gradient without dominating MLM/ITM signal
                    pw_o2r = float((n_neg_o2r / n_pos_o2r).clamp(1.0, 20.0).item())

                    valid_r2o = r2o_labels[mask]
                    n_pos_r2o = (valid_r2o > 0.5).float().sum().clamp_min(1.0)
                    n_neg_r2o = (valid_r2o == 0.0).float().sum().clamp_min(1.0)
                    pw_r2o = float((n_neg_r2o / n_pos_r2o).clamp(1.0, 20.0).item())

                loss_i = F.binary_cross_entropy_with_logits(
                    logits_per_image[mask].float(),
                    o2r_labels[mask],
                    pos_weight=torch.tensor([pw_o2r], device=logits_per_image.device, dtype=torch.float),
                    reduction="mean",
                )

                loss_t = F.binary_cross_entropy_with_logits(
                    logits_per_text[mask].float(),
                    r2o_labels[mask],
                    pos_weight=torch.tensor([pw_r2o], device=logits_per_text.device, dtype=torch.float),
                    reduction="mean",
                )

                contrastive_loss = (loss_i + loss_t) / 2

            else:
                raise RuntimeError(
                    "TWC branch ran but o2r_block has no valid labels. "
                    "Check OCR Aug label generation."
                )

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
    FIX: Balanced TWC Accuracy = (Positive Recall + Negative Recall) / 2

    Tại sao fix:
    - Ma trận label [BN×BN] có ~99.9% negative cells (label=0.0) và ~0.1% positive.
    - Metric cũ tính accuracy trên TẤT CẢ cells → trivially ~99.7% khi model
      chỉ học push tất cả logits xuống âm (predict negative cho mọi pair).
    - Metric mới đo riêng:
        * pos_recall: % positive pairs (label>0.5) được predict đúng (logit>0)
        * neg_recall: % negative pairs (label=0) được predict đúng (logit<=0)
      và báo trung bình macro → không bị dominated bởi lớp negative.
    - Để tránh noise từ việc sample ngẫu nhiên, neg_recall được tính trên
      tập negative với kích thước min(4*n_pos, n_neg) để balanced nhưng
      có đủ negative signal.
    """
    def __init__(self):
        super().__init__("twc_acc")

    def calculate(self, sample_list, model_output):
        if "contrastive_scores" not in model_output or model_output["contrastive_scores"] is None:
            return 0.0

        logits_per_image = model_output["contrastive_scores"].detach()  # [BN, BN]
        o2r_block = model_output.get("o2r_block", None)

        if o2r_block is None:
            return 0.0

        # o2r_block đã là [BN, BN] block-diag từ model output
        # Tách riêng positive và strict-negative cells
        pos_mask = (o2r_block > 0.5)                # label > 0.5 → positive pair
        neg_mask = (o2r_block == 0.0)               # label == 0 → negative pair (strict)
        # Bỏ qua cells có label = -1 (padding) và label ∈ (0, 0.5] (mềm)

        # ── Positive Recall ────────────────────────────────────────────────────
        # "Trong tất cả positive pairs, có bao nhiêu % được predict là positive?"
        if pos_mask.any():
            pos_logits = logits_per_image[pos_mask]
            pos_recall = (pos_logits > 0.0).float().mean().item()
        else:
            pos_recall = 0.0

        # ── Negative Recall (subsampled) ───────────────────────────────────────
        # "Trong các negative pairs được chọn, có bao nhiêu % được predict đúng?"
        # Subsample để tránh bias và giữ tỷ lệ pos:neg ≈ 1:4 (đủ để ổn định)
        if neg_mask.any():
            n_pos = int(pos_mask.sum().item())
            n_neg_target = min(n_pos * 4, int(neg_mask.sum().item()))
            n_neg_target = max(n_neg_target, 1)

            neg_indices = neg_mask.nonzero(as_tuple=False)   # [K, 2]
            if neg_indices.size(0) > n_neg_target:
                perm = torch.randperm(neg_indices.size(0), device=logits_per_image.device)
                neg_indices = neg_indices[perm[:n_neg_target]]

            sampled_neg_logits = logits_per_image[neg_indices[:, 0], neg_indices[:, 1]]
            neg_recall = (sampled_neg_logits <= 0.0).float().mean().item()
        else:
            neg_recall = 1.0  # Không có negative → perfect trivially

        # Balanced accuracy: macro average của pos_recall và neg_recall
        balanced_acc = (pos_recall + neg_recall) / 2.0
        return balanced_acc, pos_recall, neg_recall


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