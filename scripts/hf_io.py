"""
scripts/hf_io.py
Chọn checkpoint để predict, tải nó (chỉ file inference) và upload kết quả predict lên HF Hub.

Chạy bằng tiến trình riêng (`!python scripts/hf_io.py ...` trong Colab) thay vì import trong
kernel: setup.sh cài lại huggingface_hub sau khi kernel đã import bản có sẵn của Colab, nên
import thêm trong kernel trộn hai phiên bản (ImportError: reset_sessions ...).

  # liệt kê checkpoint (điểm val từng epoch), tải checkpoint được chọn, ghi đường dẫn ra JSON
  python scripts/hf_io.py prepare --repo USER/REPO --dir OUTPUT_DIR --ckpt best --info_out info.json
  python scripts/hf_io.py upload_predictions --repo USER/REPO --dir PRED_DIR --path_in_repo predictions --files a.csv

--ckpt nhận:
  best              model tốt nhất ở thư mục gốc (trainer.save_model sau load_best_model_at_end)
  latest            checkpoint cuối cùng
  epoch3 / 3        checkpoint của epoch 3 (save_strategy=epoch)
  checkpoint-17580  hoặc 17580 — đúng checkpoint đó

Bố cục (khớp thư mục train trên Colab):
  best          → model: OUTPUT_DIR                    | kết quả: OUTPUT_DIR
  checkpoint-N  → model: OUTPUT_DIR/checkpoint-N       | kết quả: OUTPUT_DIR/predict_checkpoint-N
Kết quả trên HF: predictions/ (best) hoặc predictions/checkpoint-N/.
"""
import argparse
import glob
import json
import os
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

# Chỉ file cần cho inference: không tải optimizer/scheduler (~3GB mỗi checkpoint).
INFERENCE_FILES = [
    "config.json", "generation_config.json", "model.safetensors",
    "special_tokens_map.json", "spiece.model", "tokenizer.json",
    "tokenizer_config.json", "added_tokens.json", "image_processor/*",
]


def _step(name):
    try:
        return int(name.split("-")[-1])
    except ValueError:
        return -1


def list_checkpoints(api, repo, out_dir, token):
    """Trả (danh sách checkpoint-N theo step, log_history, best_model_checkpoint, repo_files)."""
    files = []
    try:
        files = api.list_repo_files(repo, token=token)
    except Exception as e:
        print(f"⚠️ Không đọc được repo {repo} ({type(e).__name__}) → chỉ dùng checkpoint local.")
    names = {f.split("/")[0] for f in files if f.startswith("checkpoint-") and f.endswith("/model.safetensors")}
    names |= {os.path.basename(os.path.dirname(p))
              for p in glob.glob(os.path.join(out_dir, "checkpoint-*", "model.safetensors"))}
    cks = sorted(names, key=_step)
    hist, best = [], ""
    for ck in reversed(cks):
        local = os.path.join(out_dir, ck, "trainer_state.json")
        path = local if os.path.exists(local) else None
        if path is None and f"{ck}/trainer_state.json" in files:
            try:
                path = hf_hub_download(repo, f"{ck}/trainer_state.json", token=token)
            except Exception:
                path = None
        if path:
            st = json.load(open(path, encoding="utf-8"))
            hist, best = st.get("log_history", []), os.path.basename(st.get("best_model_checkpoint") or "")
            break
    return cks, hist, best, set(files)


def print_table(cks, hist, best):
    ev = {h["step"]: h for h in hist if "eval_em" in h or "eval_cider" in h}
    print("\nCheckpoint có sẵn (điểm trên val lúc train):")
    print(f"  {'epoch':>5}  {'checkpoint':<18} {'EM':>7} {'F1':>7} {'(EM+F1)/2':>10} {'CIDEr':>7}")
    for i, ck in enumerate(cks, 1):
        h = ev.get(_step(ck), {})
        em, f1 = h.get("eval_em"), h.get("eval_f1")
        mean = h.get("eval_em_f1", (em + f1) / 2 if em is not None and f1 is not None else None)
        fmt = lambda v, k=100: f"{k * v:.2f}" if v is not None else "-"
        print(f"  {i:>5}  {ck:<18} {fmt(em):>7} {fmt(f1):>7} {fmt(mean):>10} {fmt(h.get('eval_cider'), 1):>7}"
              f"{'   ← best (thư mục gốc)' if ck == best else ''}")


def resolve(spec, cks):
    s = str(spec).strip().lower()
    if s in ("", "best"):
        return "best"
    if s == "latest":
        if not cks:
            sys.exit("❌ Không có checkpoint-* nào.")
        return cks[-1]
    is_epoch = s.startswith("epoch")
    if is_epoch:
        s = s[5:].strip(":= ")
        if not s.isdigit() or not 1 <= int(s) <= len(cks):
            sys.exit(f"❌ '{spec}': chỉ có {len(cks)} epoch (checkpoint) để chọn.")
    if s.startswith("checkpoint-"):
        ck = s
    elif s.isdigit() and 1 <= int(s) <= len(cks):
        ck = cks[int(s) - 1]
    elif s.isdigit():
        ck = f"checkpoint-{int(s)}"
    else:
        sys.exit(f"❌ --ckpt '{spec}' không hợp lệ (best | latest | epochN | checkpoint-N).")
    if ck not in cks:
        sys.exit(f"❌ Không có {ck}. Có: {', '.join(cks) or '(không có)'}")
    return ck


def prepare(a, token):
    api = HfApi(token=token)
    cks, hist, best, files = list_checkpoints(api, a.repo, a.dir, token)
    print_table(cks, hist, best)
    ck = resolve(a.ckpt, cks)
    sub = "" if ck == "best" else ck
    model_dir = os.path.join(a.dir, sub) if sub else a.dir
    pred_dir = os.path.join(a.dir, f"predict_{ck}") if sub else a.dir
    pred_repo = f"predictions/{ck}" if sub else "predictions"
    os.makedirs(pred_dir, exist_ok=True)

    pats = [] if os.path.exists(os.path.join(model_dir, "model.safetensors")) else \
        [f"{sub}/{p}" if sub else p for p in INFERENCE_FILES]
    if any(f.startswith(pred_repo + "/") and f.count("/") == pred_repo.count("/") + 1 for f in files):
        pats.append(f"{pred_repo}/*")
    if pats:
        print(f"\n⬇️ Tải {'model' if any('model.safetensors' in p for p in pats) else 'kết quả cũ'} "
              f"[{ck}] từ {a.repo}")
        snapshot_download(repo_id=a.repo, repo_type="model", local_dir=a.dir, token=token, allow_patterns=pats)
    if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
        sys.exit(f"❌ Không có model cho '{ck}' ({model_dir}) — local lẫn {a.repo}.")
    # Kết quả predict cũ của checkpoint này (nếu đã upload) → đưa về pred_dir để chỉ chấm lại.
    for f in glob.glob(os.path.join(a.dir, *pred_repo.split("/"), "*")):
        dst = os.path.join(pred_dir, os.path.basename(f))
        if os.path.isfile(f) and not os.path.exists(dst):
            shutil.copy2(f, dst)

    info = {"ckpt": ck, "model_dir": model_dir, "pred_dir": pred_dir,
            "pred_path_in_repo": pred_repo, "attn_path_in_repo": pred_repo.replace("predictions", "attention", 1)}
    print(f"\n✅ Checkpoint dùng để predict: {ck}\n   model : {model_dir}\n   kết quả: {pred_dir}")
    if a.info_out:
        with open(a.info_out, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


def upload(a, token):
    if not token:
        sys.exit("❌ Thiếu HF_TOKEN → không upload được; hãy tải file về trước khi ngắt phiên.")
    api = HfApi(token=token)
    for f in a.files:
        p = os.path.join(a.dir, f)
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{a.path_in_repo}/{f}",
                            repo_id=a.repo, repo_type="model")
            print(f"↑ {a.repo}/{a.path_in_repo}/{f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "download_best", "upload_predictions"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", default="best", help="best | latest | epochN | checkpoint-N")
    ap.add_argument("--info_out", default="")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--path_in_repo", default="predictions")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or None
    if a.action == "download_best":
        a.ckpt = "best"
        a.action = "prepare"
    if a.action == "prepare":
        prepare(a, token)
    else:
        upload(a, token)


if __name__ == "__main__":
    main()
