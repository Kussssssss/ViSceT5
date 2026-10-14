"""Write submission files (ID,Answer) for the validation/test split of a dataset.

    python -m training.predict --ckpt_dir output/finetune --dataset ViTextVQA --split both

The related OCR copy (if the model uses it) comes from the deterministic correction
channel with the lexicon built from the TRAIN split, exactly as during evaluation.
Test answers are never read (the public test JSON may hold placeholders); dev EM/F1
use the same normalisation as the validation metric during training.
"""

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from data.dataset import ViT5VQADataset
from training.common import build_collator, build_corrector, load_dataframes, load_model_and_tokenizer
from training.metrics import compute_f1_em

SPLITS = {"dev": "validation", "test": "test"}


def _clean(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


@torch.no_grad()
def predict_split(model, tokenizer, collator, frame, batch_size, num_beams, max_new_tokens, labelled):
    device = next(model.parameters()).device
    frame = frame.copy()
    if not labelled:
        frame["answer"] = ""
    dataset = ViT5VQADataset(frame)
    predictions = []
    for start in range(0, len(dataset), batch_size):
        batch = collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        batch.pop("labels", None)
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        tokens = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
        predictions += tokenizer.batch_decode(tokens, skip_special_tokens=True)
        done = min(start + batch_size, len(dataset))
        if (start // batch_size) % 50 == 0 or done == len(dataset):
            print(f"  {done}/{len(dataset)}", flush=True)
    return predictions


def main(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--dataset", default="ViTextVQA")
    parser.add_argument("--data_dir", default="./datasets")
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument("--out_dir", default="./output")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=0, help="0 = config value")
    parser.add_argument("--max_new_tokens", type=int, default=0, help="0 = config value")
    args = parser.parse_args(args_list)

    model, tokenizer = load_model_and_tokenizer(args.ckpt_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    frames = load_dataframes(args.dataset, args.data_dir)
    corrector = build_corrector(model.config, frames["train"]) if model.config.ablation_use_ocr_aug else None
    collator = build_collator(model, tokenizer, corrector, training=False)
    beams = args.num_beams or model.config.generation_num_beams
    max_new = args.max_new_tokens or model.config.generation_max_new_tokens
    tag = "" if args.dataset == "ViTextVQA" else f"{args.dataset}_"
    outputs = []
    for split in (("dev", "test") if args.split == "both" else (args.split,)):
        frame = frames[SPLITS[split]]
        if frame.empty or "id" not in frame:
            print(f"[{split}] no data or no 'id' column — skipped.")
            continue
        print(f"[{split}] {len(frame)} questions | beams={beams} max_new_tokens={max_new}")
        answers = predict_split(model, tokenizer, collator, frame, args.batch_size, beams, max_new,
                                labelled=split == "dev")
        path = Path(args.out_dir) / f"submission_{tag}{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": frame["id"].tolist(), "Answer": [_clean(a) for a in answers]}).to_csv(
            path, index=False, encoding="utf-8"
        )
        print(f"Wrote {len(answers)} rows -> {path}")
        if split == "dev":
            f1, em = compute_f1_em(answers, frame["answer"].fillna("").astype(str).tolist())
            print(f"📊 [dev] EM={em:.4f} ({em * 100:.2f}%) | F1={f1:.4f}")
        outputs.append(str(path))
    return outputs


if __name__ == "__main__":
    main()
