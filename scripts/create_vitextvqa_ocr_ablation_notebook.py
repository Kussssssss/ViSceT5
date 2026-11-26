"""Build the ViTextVQA OCR ablation Colab notebook from the ABL1 workflow."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/ablations/ViTextVQA/ABL1_baseline_bs16x1.ipynb"
DEST = SOURCE.with_name("ABL_OCR_features_bs16x1.ipynb")
BRANCH = "exp/vitextvqa-ocr-ablation"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}, got {source.count(old)}")
    return source.replace(old, new, 1)


def main():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [
        "# ViTextVQA — ablation đặc trưng OCR\n",
        "Chọn kịch bản 1, 2, 3 hoặc 5 trong cell cấu hình. Kịch bản 4 là ABL1. "
        "Notebook dùng nhánh exp/vitextvqa-ocr-ablation, cấu hình ABL1 "
        "16×1, LR 1e-4, QA-ViT/Consformer/TWC tắt. Code này khác commit ABL1 lịch sử; "
        "cần chạy thêm kịch bản 4 trên cùng code để so sánh có kiểm soát.\n",
        "Mỗi kịch bản có output và Hugging Face repo riêng. Chạy cell theo thứ tự.\n",
    ]
    config = "".join(notebook["cells"][4]["source"])
    config = replace_once(config, 'GIT_COMMIT     = "cc773b0"', f'GIT_COMMIT     = "{BRANCH}"')
    config = replace_once(config, 'QAVIT          = "off"',
        'QAVIT          = "off"\nOCR_SCENARIO = 1  # 1=text; 2=text+char; 3=text+char+bbox; 5=unaligned; 4=full control')
    config = replace_once(config, 'PREDICT_MIN_DEV_EM = 31.0', 'PREDICT_MIN_DEV_EM = 0.0')
    config = replace_once(config, 'RUN_NAME += f"-bs{BATCH_SIZE}x{GRAD_ACCUM}"',
        'assert OCR_SCENARIO in (1, 2, 3, 4, 5)\nRUN_NAME += f"-ocr{OCR_SCENARIO}-bs{BATCH_SIZE}x{GRAD_ACCUM}"')
    notebook["cells"][4]["source"] = config.splitlines(keepends=True)
    setup = "".join(notebook["cells"][6]["source"])
    setup = replace_once(setup, '!git clone -q {GIT_REPO} {REPO_DIR}',
                         '!git clone -q --branch {GIT_COMMIT} {GIT_REPO} {REPO_DIR}')
    marker = 'os.environ.update(ENV, OUTPUT_DIR=OUTPUT_DIR, HF_REPO=HF_REPO, PYTHONUNBUFFERED="1")'
    setup = replace_once(setup, marker, '''ENV.update({
    "OCR_USE_CHAR": _b(OCR_SCENARIO != 1),
    "OCR_USE_BBOX": _b(OCR_SCENARIO in (3, 4, 5)),
    "OCR_USE_VISUAL": _b(OCR_SCENARIO in (4, 5)),
    "OCR_ALIGN_SPATIAL_VISUAL": _b(OCR_SCENARIO != 5),
})
assert all(k in READ for k in ENV), "Commit does not expose every requested flag"
''' + marker)
    notebook["cells"][6]["source"] = setup.splitlines(keepends=True)
    # Keep the ABL1 setup/test/train/predict flow; the old cross-run analysis cells
    # assume historical code and are not appropriate for this experiment.
    notebook["cells"] = notebook["cells"][:16]
    # Predict in a fresh Python process: the Colab kernel has imported HF packages
    # before setup.sh can upgrade/downgrade them. Mixing those cached modules with
    # files from the new installation causes errors such as missing reset_sessions.
    prediction = "".join(notebook["cells"][15]["source"])
    prediction = replace_once(prediction, '"evaluate", "bert-score", "git+https://github.com/salaniz/pycocoevalcap"], check=False)',
                              '"evaluate", "bert-score", "git+https://github.com/salaniz/pycocoevalcap"], check=True)')
    prediction = replace_once(prediction,
        'probe = ("import huggingface_hub, transformers, evaluate, bert_score, pycocoevalcap;"\n'
        '             "from transformers.models.auto.modeling_auto import MODEL_MAPPING; list(MODEL_MAPPING.keys())")',
        'probe = "from huggingface_hub import HfApi, hf_hub_download; import transformers, evaluate, bert_score, pycocoevalcap"')
    import base64
    encoded = base64.b64encode(prediction.encode("utf-8")).decode("ascii")
    wrapper = f'''# Dự đoán trong Python mới để không dùng module Hugging Face đã cache trong kernel.
import json, os, subprocess, sys
_prediction_env = os.environ.copy()
_prediction_env["VISCET5_PREDICTION_CONTEXT"] = json.dumps({{
    key: globals()[key] for key in ("CODE_SHA", "OUTPUT_DIR", "DATASET", "HF_REPO",
                                     "PREDICT_CKPT", "PREDICT_MIN_DEV_EM")
}})
_bootstrap = """
import base64, json, os
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
shell.user_ns.update(json.loads(os.environ['VISCET5_PREDICTION_CONTEXT']))
source = base64.b64decode('{encoded}').decode('utf-8')
code = shell.transform_cell(source)
exec(compile(code, '<predict>', 'exec'), shell.user_ns)
"""
subprocess.run([sys.executable, "-u", "-c", _bootstrap], env=_prediction_env, check=True)
'''
    notebook["cells"][15]["source"] = wrapper.splitlines(keepends=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()
