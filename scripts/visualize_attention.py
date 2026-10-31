"""
scripts/visualize_attention.py
Trực quan hoá attention của ViSceT5 khi sinh câu trả lời.

Đầu vào encoder là chuỗi [câu hỏi | 196 patch ảnh CLIP (nếu có ảnh) | token OCR gốc |
token OCR đã sửa (nếu bật OCR-Aug)]. Script sinh đáp án (beam search, giống predict.py),
rồi chạy lại decoder theo đúng đáp án đó để lấy CROSS-ATTENTION từ các token đáp án tới
từng vị trí của encoder (trung bình trên các lớp, các head và các token đáp án).

Mỗi mẫu ra một hình gồm:
  • ảnh + heatmap attention trên 14×14 patch CLIP (bỏ qua nếu mô hình không dùng ảnh);
  • ảnh + box của từ OCR, tô màu theo attention (gốc + bản sửa của cùng từ);
  • attention trên từng từ của câu hỏi, của OCR gốc và của OCR đã sửa (từ bị sửa có viền đỏ).
Kèm segment_mass.csv: tỷ lệ attention dồn vào câu hỏi / ảnh / OCR gốc / OCR sửa cho mỗi mẫu.

Cách dùng:
    python scripts/visualize_attention.py --ckpt_dir output/finetune \
        --dataset ViTextVQA --split test --n 8 --gt eval_gt/ViTextVQA_test_gt.json \
        --out_dir output/finetune/attention
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches
from PIL import Image
from transformers.modeling_outputs import BaseModelOutput

from data.dataset import ViT5VQADataset
from models.openvivqa_model import RELATED_FIELDS
from training.common import build_collator, build_corrector, load_dataframes, load_model_and_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_KEYS = ["input_ids", "attention_mask", "pixel_values", "ocr_info", "ocr_input_ids",
              "ocr_attention_mask", "ocr_to_word_map", "ocr_char_ids", "ocr_char_mask", "ocr_word_mask",
              *RELATED_FIELDS]
CMAP = matplotlib.colormaps["YlOrRd"]


def _to_dev(x):
    return x.to(DEVICE) if isinstance(x, torch.Tensor) else x


@torch.inference_mode()
def answer_cross_attention(model, batch, num_beams, max_new_tokens, layer):
    """Sinh đáp án rồi trả (đáp án, vector attention trên toàn bộ vị trí encoder)."""
    inp = {k: _to_dev(batch[k]) for k in MODEL_KEYS if k in batch}
    enc_out, enc_mask, _ = model.encode(**inp)
    # generate (beam search) nhân bản encoder_outputs NGAY TRÊN object truyền vào → đưa bản sao.
    seq = model.vit5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=enc_out.last_hidden_state.clone()),
                              attention_mask=enc_mask.clone(), use_cache=False,
                              max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False,
                              pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id,
                              decoder_start_token_id=model.config.decoder_start_token_id)[0]
    ids = seq.tolist()
    start = model.config.decoder_start_token_id
    if not ids or ids[0] != start:
        ids = [start] + ids
    eos = model.config.eos_token_id
    k = ids.index(eos, 1) if eos in ids[1:] else len(ids)
    answer_ids = ids[1:k]
    dec_in = torch.tensor([ids[:max(k, 1)]], device=DEVICE)
    out = model.vit5(encoder_outputs=enc_out, attention_mask=enc_mask, decoder_input_ids=dec_in,
                     output_attentions=True, return_dict=True, use_cache=False)
    xs = out.cross_attentions if layer == "all" else (out.cross_attentions[-1],)
    att = torch.stack([x[0].float().mean(0) for x in xs]).mean(0)   # (T_dec, S)
    rows = att[: max(len(answer_ids), 1)]                            # hàng sinh ra từng token đáp án
    vec = rows.mean(0).cpu().numpy()
    return answer_ids, vec, enc_mask[0].cpu().numpy()


def merge_pieces(tokens, weights, mask):
    """Gộp sub-word SentencePiece ('▁' mở đầu từ) thành từ, cộng attention."""
    words, ws = [], []
    for t, w, m in zip(tokens, weights, mask):
        if not m or t in ("</s>", "<pad>"):
            continue
        if t.startswith("▁") or not words:
            words.append(t.lstrip("▁"))
            ws.append(float(w))
        else:
            words[-1] += t
            ws[-1] += float(w)
    keep = [(a, b) for a, b in zip(words, ws) if a]
    return [a for a, _ in keep], [b for _, b in keep]


def draw_chips(ax, title, words, weights, vmax, changed=None, max_words=90):
    """Vẽ các từ thành 'chip' tô màu theo attention, tự xuống dòng."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, title, fontsize=10, fontweight="bold", va="top")
    if not words:
        ax.text(0.0, 0.75, "(không có)", fontsize=9, va="top", color="gray")
        return
    extra = len(words) - max_words
    words, weights = words[:max_words], weights[:max_words]
    changed = (changed or [False] * len(words))[:max_words]
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    x, y, line_h = 0.0, 0.78, 0.2
    for w, a, ch in zip(words, weights, changed):
        col = CMAP(min(a / vmax, 1.0) if vmax > 0 else 0.0)
        t = ax.text(x, y, w, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="red" if ch else "none", lw=1.2))
        bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        if bb.x1 > 1.0 and x > 0:
            t.remove()
            x, y = 0.0, y - line_h
            t = ax.text(x, y, w, fontsize=9, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="red" if ch else "none", lw=1.2))
            bb = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        x = bb.x1 + 0.008
    if extra > 0:
        ax.text(x, y, f"… (+{extra} từ)", fontsize=8, va="top", color="gray")


def visualize(qid, sample, batch, tokenizer, answer, gt, vec, enc_mask, out_png):
    L_q = batch["input_ids"].size(1)
    related = "ocr_rel_input_ids" in batch  # OCR block = [original OCR | related OCR] (TWC)
    L_ocr = batch["ocr_input_ids"].size(1) + (batch["ocr_rel_input_ids"].size(1) if related else 0)
    S = vec.shape[0]
    L_img = S - L_q - L_ocr
    total = float(vec.sum()) or 1.0

    # câu hỏi
    q_tok = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
    q_words, q_w = merge_pieces(q_tok, vec[:L_q], enc_mask[:L_q])

    # ảnh
    img_vec = vec[L_q:L_q + L_img] if L_img > 0 else None

    # OCR: token → từ (ocr_to_word_map), bản gốc ở slot 0..W-1, bản related ở W..2W-1
    ocr_vec = vec[L_q + L_img:]
    W_slots = batch["ocr_char_ids"].shape[1]
    wmap = batch["ocr_to_word_map"][0].tolist()
    if related:
        wmap += [w + W_slots if w >= 0 else -1 for w in batch["ocr_rel_to_word_map"][0].tolist()]
    words = batch["ocr_words"][0]
    orig, rel = words["orig"], words["rel"]
    n = len(orig)  # entry OCR thật (không gồm PAD)
    slots = W_slots * (2 if related else 1)
    half = slots // 2 if rel is not None else slots
    w_att = np.zeros(slots)
    for t, wi in enumerate(wmap[:len(ocr_vec)]):
        if 0 <= wi < slots:
            w_att[wi] += ocr_vec[t]
    real = np.ones(n, dtype=bool)
    o_att = w_att[:n]
    r_att = w_att[half:half + n] if rel is not None else np.zeros(n)
    o_words = [w for w, m in zip(orig, real) if m]
    o_w = [float(a) for a, m in zip(o_att, real) if m]
    r_words = [w for w, m in zip(rel, real) if m] if rel is not None else []
    r_w = [float(a) for a, m in zip(r_att, real) if m] if rel is not None else []
    r_changed = [a != b for a, b in zip(r_words, o_words)]

    mass = {"question": float(np.sum(q_w)) / total,
            "image": float(img_vec.sum()) / total if img_vec is not None else 0.0,
            "ocr_orig": float(o_att[real].sum()) / total,
            "ocr_rel": float(r_att[real].sum()) / total if rel is not None else 0.0}

    image = Image.open(sample["image_path"]).convert("RGB")
    W, H = image.size
    has_img = img_vec is not None and L_img > 0
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[4.2, 1.2, 1.6, 1.6], hspace=0.3, wspace=0.05)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image)
    if has_img:
        g = int(round(math.sqrt(L_img)))
        heat = img_vec[: g * g].reshape(g, g)
        heat = heat / (heat.max() or 1.0)
        heat_im = Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        ax_img.imshow(np.asarray(heat_im) / 255.0, cmap="jet", alpha=0.45, extent=(0, W, H, 0))
        ax_img.set_title(f"Attention trên {g}×{g} patch ảnh ({100 * mass['image']:.1f}% tổng)", fontsize=10)
    else:
        ax_img.set_title("Mô hình không dùng đặc trưng ảnh (CLIP đã bỏ)", fontsize=10)
    ax_img.axis("off")

    ax_box = fig.add_subplot(gs[0, 1])
    ax_box.imshow(image)
    boxes = batch["ocr_info"][0]["boxes"].cpu().numpy()[:n]
    box_att = o_att + r_att
    vmax_box = float(box_att[real].max()) if real.any() else 1.0
    top = set(np.argsort(-np.where(real, box_att, -1))[:3].tolist())
    for j, (b, a, m) in enumerate(zip(boxes, box_att, real)):
        if not m:
            continue
        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        col = CMAP(0.15 + 0.85 * min(a / vmax_box, 1.0) if vmax_box > 0 else 0.15)
        ax_box.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec=col, fc=col, alpha=0.5))
        if j in top:
            ax_box.text(x1, max(y1 - 4, 0), orig[j], fontsize=8, color="black", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1.5, alpha=0.9))
    ax_box.set_title(f"Attention trên từ OCR (gốc + related, {100 * (mass['ocr_orig'] + mass['ocr_rel']):.1f}% tổng)",
                     fontsize=10)
    ax_box.axis("off")

    head = fig.add_subplot(gs[1, :])
    head.axis("off")
    q_text = " ".join(q_words)
    short = lambda t, k=110: t if len(t) <= k else t[:k] + "…"
    head.text(0, 1.0, short(f"Câu hỏi: {q_text}", 150), fontsize=11, va="top")
    head.text(0, 0.62, f"Dự đoán: {short(answer)}    |    Đáp án: {short(str(gt)) if gt else '(không có)'}",
              fontsize=11, va="top",
              color="green" if gt and answer.strip().lower() == str(gt).strip().lower() else "black")
    head.text(0, 0.0, "Tỷ lệ attention — câu hỏi {:.1f}% · ảnh {:.1f}% · OCR gốc {:.1f}% · OCR related {:.1f}%".format(
        100 * mass["question"], 100 * mass["image"], 100 * mass["ocr_orig"], 100 * mass["ocr_rel"]),
        fontsize=10, va="bottom", color="dimgray")

    vmax_txt = max([*q_w, *o_w, *r_w, 1e-9])
    draw_chips(fig.add_subplot(gs[2, 0]), "Câu hỏi", q_words, q_w, vmax_txt)
    draw_chips(fig.add_subplot(gs[2, 1]), "OCR gốc", o_words, o_w, vmax_txt)
    draw_chips(fig.add_subplot(gs[3, :]), "OCR related (viền đỏ = từ bị đổi)" if rel is not None else "OCR related (TWC tắt)",
               r_words, r_w, vmax_txt, changed=r_changed)
    sm = cm.ScalarMappable(norm=colors.Normalize(0, vmax_txt), cmap=CMAP)
    fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.01, label="attention (cùng thang cho câu hỏi và OCR)")
    fig.suptitle(f"ID {qid}", fontsize=12)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", default="ViTextVQA")
    ap.add_argument("--data_dir", default="./datasets")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--ids", nargs="*", type=int, default=[], help="ID câu hỏi cần vẽ (bỏ trống = lấy ngẫu nhiên)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt", default="", help="JSON đáp án (để hiện đáp án đúng trên tập test)")
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--layer", default="all", choices=["all", "last"], help="trung bình mọi lớp decoder hay chỉ lớp cuối")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.ckpt_dir)
    model.to(DEVICE).eval()

    frames = load_dataframes(args.dataset, args.data_dir)
    df = frames["validation" if args.split == "dev" else "test"].copy()
    if args.split == "test":
        df["answer"] = ""
    gt_map = {}
    if args.gt and os.path.exists(args.gt):
        with open(args.gt, encoding="utf-8") as f:
            gt_map = {int(a["id"]): a["answers"][0] for a in json.load(f)["annotations"]}
    if args.ids:
        df = df[df["id"].astype(int).isin(args.ids)]
    else:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    corrector = build_corrector(model.config, frames["train"]) if model.config.use_twc else None
    collator = build_collator(model, tokenizer, corrector, training=False)
    collator.return_ocr_words = True
    max_new = int(getattr(model.config, "generation_max_new_tokens", 56))

    df = df.reset_index(drop=True)
    ds = ViT5VQADataset(df)
    rows = []
    for i in range(len(ds)):
        sample, qid = ds[i], int(df.iloc[i]["id"])
        batch = collator([sample])
        answer_ids, vec, enc_mask = answer_cross_attention(model, batch, args.num_beams, max_new, args.layer)
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
        gt = gt_map.get(qid) or sample.get("answer") or ""
        out_png = os.path.join(args.out_dir, f"attn_{args.split}_{qid}.png")
        mass = visualize(qid, sample, batch, tokenizer, answer, gt, vec, enc_mask, out_png)
        rows.append({"id": qid, "question": sample["question"], "pred": answer, "gt": gt, **mass})
        print(f"✅ {out_png} | dự đoán='{answer}' đáp án='{gt}' | " +
              " ".join(f"{k}={100 * v:.1f}%" for k, v in mass.items()))

    with open(os.path.join(args.out_dir, "segment_mass.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = ["question", "image", "ocr_orig", "ocr_rel"]
    print("Trung bình tỷ lệ attention:", {k: round(100 * float(np.mean([r[k] for r in rows])), 1) for k in keys})


if __name__ == "__main__":
    main()
