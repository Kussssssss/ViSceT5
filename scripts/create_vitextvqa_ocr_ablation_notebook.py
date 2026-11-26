"""Build a self-contained Colab notebook from the ViTextVQA ABL1 workflow.

Run from the repository root while the OCR feature changes are in the working
tree. The notebook applies those changes to the pinned main commit it clones.
"""
import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/ablations/ViTextVQA/ABL1_baseline_bs16x1.ipynb"
DEST = SOURCE.with_name("ABL_OCR_features_bs16x1.ipynb")
FILES = ["configs/arguments.py", "configs/model_config.py", "models/modules/ocr_spatial.py",
         "models/openvivqa_model.py", "run_pipeline.py", "training/finetune.py",
         "tests/test_vqa_pipeline.py"]
BASE = "8842c49"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}, got {source.count(old)}")
    return source.replace(old, new, 1)


def main():
    patch = subprocess.check_output(["git", "diff", "--binary", BASE, "--", *FILES], cwd=ROOT)
    if not patch:
        raise RuntimeError("No OCR ablation patch found")
    encoded = base64.b64encode(patch).decode("ascii")
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# ViTextVQA — ablation đặc trưng OCR\n",
        "Chọn kịch bản 1, 2, 3 hoặc 5 trong cell cấu hình. Kịch bản 4 là ABL1. "
        "Notebook dùng code main 8842c49 cùng patch OCR được nhúng, cấu hình ABL1 "
        "16×1, LR 1e-4, QA-ViT/Consformer/TWC tắt. Code này khác commit ABL1 lịch sử; "
        "cần chạy thêm kịch bản 4 trên cùng code để so sánh có kiểm soát.\n",
        "Mỗi kịch bản có output và Hugging Face repo riêng. Chạy cell theo thứ tự.\n",
    ]
    config = "".join(notebook["cells"][4]["source"])
    config = replace_once(config, 'GIT_COMMIT     = "cc773b0"', f'GIT_COMMIT     = "{BASE}"')
    config = replace_once(config, 'QAVIT          = "off"',
        'QAVIT          = "off"\nOCR_SCENARIO = 1  # 1=text; 2=text+char; 3=text+char+bbox; 5=unaligned; 4=full control')
    config = replace_once(config, 'PREDICT_MIN_DEV_EM = 31.0', 'PREDICT_MIN_DEV_EM = 0.0')
    config = replace_once(config, 'RUN_NAME += f"-bs{BATCH_SIZE}x{GRAD_ACCUM}"',
        'assert OCR_SCENARIO in (1, 2, 3, 4, 5)\nRUN_NAME += f"-ocr{OCR_SCENARIO}-bs{BATCH_SIZE}x{GRAD_ACCUM}"')
    notebook["cells"][4]["source"] = config.splitlines(keepends=True)
    setup = "".join(notebook["cells"][6]["source"])
    marker = 'subprocess.run(["git", "checkout", "-q", "--detach", CODE_SHA], check=True)'
    apply = marker + f'''
import base64
_patch = base64.b64decode("{encoded}")
_check = subprocess.run(["git", "apply", "--check", "-"], input=_patch, capture_output=True)
if _check.returncode == 0:
    subprocess.run(["git", "apply", "-"], input=_patch, check=True)
else:
    _reverse = subprocess.run(["git", "apply", "--reverse", "--check", "-"], input=_patch, capture_output=True)
    if _reverse.returncode != 0:
        raise RuntimeError("Cannot apply embedded OCR patch: " + _check.stderr.decode(errors="replace"))
print("Applied embedded OCR ablation patch to", CODE_SHA)
'''
    setup = replace_once(setup, marker, apply)
    marker = 'os.environ.update(ENV, OUTPUT_DIR=OUTPUT_DIR, HF_REPO=HF_REPO, PYTHONUNBUFFERED="1")'
    setup = replace_once(setup, marker, '''ENV.update({
    "OCR_USE_CHAR": _b(OCR_SCENARIO != 1),
    "OCR_USE_BBOX": _b(OCR_SCENARIO in (3, 4, 5)),
    "OCR_USE_VISUAL": _b(OCR_SCENARIO in (4, 5)),
    "OCR_ALIGN_SPATIAL_VISUAL": _b(OCR_SCENARIO != 5),
})
assert all(k in READ for k in ENV), "Commit/patched pipeline does not expose every requested flag"
''' + marker)
    notebook["cells"][6]["source"] = setup.splitlines(keepends=True)
    # Keep the ABL1 setup/test/train/predict flow; the old cross-run analysis cells
    # assume historical code and are not appropriate for this experiment.
    notebook["cells"] = notebook["cells"][:16]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()
