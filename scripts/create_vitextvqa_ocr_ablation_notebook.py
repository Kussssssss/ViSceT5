"""Build the ViTextVQA OCR ablation Colab notebook from the ABL1 workflow."""
import copy
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
        "Chọn kịch bản 1–6 trong cell cấu hình. Kịch bản 4 là ABL1; kịch bản 6 giữ "
        "subword đầu tiên của mỗi entry OCR, cùng character/bbox/det/rec của entry đó. "
        "Đây là embedding đầu vào ViT5, không phải hidden state BERT của paper TWA. "
        "Notebook dùng nhánh exp/vitextvqa-ocr-ablation, cấu hình ABL1 "
        "16×1, LR 1e-4, QA-ViT/Consformer/TWC tắt. Code này khác commit ABL1 lịch sử; "
        "cần chạy thêm kịch bản 4 trên cùng code để so sánh có kiểm soát.\n",
        "Mỗi kịch bản có output và Hugging Face repo riêng; run hoàn tất được tự nhận diện "
        "và bỏ qua train. Tên run chứa learning rate: đổi LR tạo repo và output riêng. "
        "Repo cũ không ghi LR được giữ nguyên, không tự dùng cho run mới.\n",
    ]
    config = "".join(notebook["cells"][4]["source"])
    config = replace_once(config, 'GIT_COMMIT     = "cc773b0"', f'GIT_COMMIT     = "{BRANCH}"')
    config = replace_once(config, 'QAVIT          = "off"',
        'QAVIT          = "off"\nOCR_SCENARIO = 1  # 1=text; 2=+char; 3=+bbox; 4=full; 5=unaligned; 6=first subword, word-aligned')
    config = replace_once(config, 'PREDICT_MIN_DEV_EM = 31.0', 'PREDICT_MIN_DEV_EM = 0.0')
    config = replace_once(config, 'RUN_NAME += f"-bs{BATCH_SIZE}x{GRAD_ACCUM}"',
        'assert OCR_SCENARIO in (1, 2, 3, 4, 5, 6)\n'
        '_lr_mantissa, _lr_exponent = f"{LEARNING_RATE:.12e}".split("e")\n'
        'LR_TAG = _lr_mantissa.rstrip("0").rstrip(".") + "e" + str(int(_lr_exponent))\n'
        'RUN_NAME += f"-ocr{OCR_SCENARIO}-bs{BATCH_SIZE}x{GRAD_ACCUM}-lr{LR_TAG}"')
    notebook["cells"][4]["source"] = config.splitlines(keepends=True)
    setup = "".join(notebook["cells"][6]["source"])
    setup = replace_once(setup, 'import os, re, subprocess, yaml', 'import os, re, subprocess, sys, yaml')
    setup = replace_once(setup, '!git clone -q {GIT_REPO} {REPO_DIR}',
                         '!git clone -q --branch {GIT_COMMIT} {GIT_REPO} {REPO_DIR}')
    setup = replace_once(setup,
        'OUTPUT_DIR = f"/content/ViSceT5/output/{CODE_TAG}_{DATASET}_{RUN_NAME}"\n'
        'HF_REPO = f"{HF_USER}/ViSceT5-{CODE_TAG}-{DATASET.lower()}-{RUN_NAME}"',
        '# Stable experiment namespace; 9aafc6b is the original published ablation code.\n'
        'RUN_TAG = "9aafc6b"\n'
        'OUTPUT_DIR = f"/content/ViSceT5/output/{RUN_TAG}_{DATASET}_{RUN_NAME}"\n'
        'HF_REPO = f"{HF_USER}/ViSceT5-{RUN_TAG}-{DATASET.lower()}-{RUN_NAME}"')
    marker = 'os.environ.update(ENV, OUTPUT_DIR=OUTPUT_DIR, HF_REPO=HF_REPO, PYTHONUNBUFFERED="1")'
    setup = replace_once(setup, marker, '''ENV.update({
    "OCR_USE_CHAR": _b(OCR_SCENARIO != 1),
    "OCR_USE_BBOX": _b(OCR_SCENARIO in (3, 4, 5, 6)),
    "OCR_USE_VISUAL": _b(OCR_SCENARIO in (4, 5, 6)),
    "OCR_ALIGN_SPATIAL_VISUAL": _b(OCR_SCENARIO != 5),
    "OCR_TOKENIZATION": "first_subword" if OCR_SCENARIO == 6 else "all_subwords",
})
assert all(k in READ for k in ENV), "Commit does not expose every requested flag"
''' + marker + '''
# Check final Trainer state in a fresh process, avoiding stale HF modules after setup.
# A failed lookup is an error, not permission to silently start a new run.
_status_script = """
import json, os, sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
repo = os.environ['HF_REPO']
token = os.environ.get('HF_TOKEN') or None
try:
    files = HfApi(token=token).list_repo_files(repo_id=repo, token=token)
except RepositoryNotFoundError:
    sys.exit(3)
has_model = any(f in files for f in ('model.safetensors', 'model.safetensors.index.json',
                                      'pytorch_model.bin', 'pytorch_model.bin.index.json'))
states = [f for f in files if f.startswith('checkpoint-') and f.endswith('/trainer_state.json')]
states.sort(key=lambda f: int(f.split('/')[0].split('-')[-1]))
state_path = 'trainer_state.json' if 'trainer_state.json' in files else (states[-1] if states else None)
if not state_path:
    sys.exit(4)
state = json.load(open(hf_hub_download(repo, state_path, token=token), encoding='utf-8'))
steps, maximum = state.get('global_step', 0), state.get('max_steps', 0)
if not maximum or steps < maximum:
    sys.exit(4)
if not has_model:
    best = (state.get('best_model_checkpoint') or '').replace(chr(92), '/').split('/')[-1]
    if not best or not any(f.startswith(best + '/model.') or f == best + '/pytorch_model.bin' for f in files):
        sys.exit(4)
    print(best)
sys.exit(0)
"""
_runs = []
for _tag in ("9aafc6b", "a5df1da"):
    _repo = f"{HF_USER}/ViSceT5-{_tag}-{DATASET.lower()}-{RUN_NAME}"
    _env = dict(os.environ, HF_REPO=_repo)
    _status = subprocess.run([sys.executable, "-c", _status_script], env=_env, capture_output=True, text=True)
    if _status.returncode not in (0, 3, 4):
        print(_status.stderr)
        raise RuntimeError("Không kiểm tra được checkpoint trên Hugging Face; dừng để tránh train nhầm.")
    if _status.returncode in (0, 4):
        _runs.append((_status.returncode, _tag, _repo, _status.stdout.strip()))
_selected = next((run for run in _runs if run[0] == 0), _runs[0] if _runs else None)
TRAIN_FINISHED = bool(_selected and _selected[0] == 0)
if _selected:
    _, RUN_TAG, HF_REPO, _saved_best = _selected
    if _saved_best and PREDICT_CKPT == "best":
        PREDICT_CKPT = _saved_best
    OUTPUT_DIR = f"/content/ViSceT5/output/{RUN_TAG}_{DATASET}_{RUN_NAME}"
    os.environ.update(HF_REPO=HF_REPO, OUTPUT_DIR=OUTPUT_DIR)
print("Run đã hoàn tất → bỏ qua train." if TRAIN_FINISHED else "Chưa có run hoàn tất → train hoặc resume.")
print(f"Đã chọn HF_REPO={HF_REPO}")
''')
    notebook["cells"][6]["source"] = setup.splitlines(keepends=True)
    # Keep the ABL1 setup/test/train/predict flow; the old cross-run analysis cells
    # assume historical code and are not appropriate for this experiment.
    notebook["cells"] = notebook["cells"][:16]
    for idx in (8, 9, 11):
        old = "".join(notebook["cells"][idx]["source"])
        wrapped = ('if TRAIN_FINISHED:\n'
                   '    print("Run đã hoàn tất trên Hugging Face; bỏ qua bước này.")\n'
                   'else:\n' + ''.join('    ' + line if line.strip() else line
                                        for line in old.splitlines(keepends=True)))
        notebook["cells"][idx]["source"] = wrapped.splitlines(keepends=True)
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
    run_all = copy.deepcopy(notebook)
    variant_config = "".join(run_all["cells"][4]["source"])
    variant_config = replace_once(variant_config, 'BATCH_SIZE     = 16', 'BATCH_SIZE     = 8')
    variant_config = replace_once(variant_config, 'GRAD_ACCUM     = 1', 'GRAD_ACCUM     = 2')
    run_all["cells"][4]["source"] = variant_config.splitlines(keepends=True)
    run_all["cells"][0]["source"] = [s.replace("16×1", "8×2") for s in run_all["cells"][0]["source"]]
    # The manual checkpoint-upload cell is an optional recovery operation, not a Run all step.
    run_all["cells"] = [cell for i, cell in enumerate(run_all["cells"]) if i not in (12, 13)]
    run_all_path = DEST.with_name("ABL_OCR_features_bs8x2_run_all.ipynb")
    if run_all_path.exists():
        run_all_path.write_text(json.dumps(run_all, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(run_all_path)


if __name__ == "__main__":
    main()
