import sys
import os
import argparse
import subprocess
import importlib

# Đảm bảo dự án nằm trong PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run():
    try:
        print(">>> [Step 1] Preparing Dataset...")
        try:
            from scripts import prepare_dataset
            args = argparse.Namespace(
                config="configs/data/ViTextVQA.yaml",
                data_dir="./datasets"
            )
            prepare_dataset.main(args)
            print(">>> Dataset preparation successful.")
        except Exception as e:
            print(f"❌ Error preparing dataset: {e}")
            raise e

        print("\n>>> [Step 2] Initializing Model & Downloading Pretrained Weights...")
        try:
            sys.argv = ["init_model.py"]
            from scripts import init_model
            init_model.main()
            print(">>> Model initialization successful.")
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise e

        stage = os.environ.get("STAGE", "finetune").lower()
        if stage == "pretrain":
            print("\n>>> [Step 3] Starting Pretrain Training...")
            try:
                sys.argv = ["pretrain.py", "configs/pretrain.yaml"]

                # Cho phép chỉnh số epoch qua env NUM_TRAIN_EPOCHS (dùng cho full run;
                # mock bỏ qua epoch vì max_steps được ưu tiên).
                _ep = os.environ.get("NUM_TRAIN_EPOCHS", "").strip()
                if _ep:
                    sys.argv.extend(["--num_train_epochs", _ep])
                    print(f">>> [pretrain] num_train_epochs = {_ep}")

                # Tiếp tục train:
                #  RESUME_FROM_CHECKPOINT = đường dẫn LOCAL tới thư mục checkpoint-XXXX đầy đủ
                #        (có optimizer/scheduler/trainer_state) -> resume ĐÚNG bước/epoch/LR.
                #  RESUME_CHECKPOINT_ID   = Drive id của zip checkpoint đầy đủ -> tải rồi resume đúng.
                #  MODEL_NAME_OR_PATH     = thư mục chỉ có trọng số (vd model tải từ HF) -> WARM-START
                #        (optimizer/LR khởi tạo lại).
                for _rk, _rflag in [("RESUME_FROM_CHECKPOINT", "--resume_from_checkpoint"),
                                    ("RESUME_CHECKPOINT_ID", "--resume_checkpoint_id"),
                                    ("MODEL_NAME_OR_PATH", "--model_name_or_path")]:
                    _rv = os.environ.get(_rk, "").strip()
                    if _rv:
                        sys.argv.extend([_rflag, _rv])
                        print(f">>> [pretrain] {_rflag} = {_rv}")

                # Chọn mục tiêu pretrain qua env LOSS_ABLATION_MODE:
                #   all (mặc định) | only_twc_ocr_aug | only_itm_mlm |
                #   gen_all (decoder read-scene-text + MLM/ITM/TWC phụ trợ) | gen (+ MLM/ITM)
                _mode = os.environ.get("LOSS_ABLATION_MODE", "").strip()
                if _mode:
                    sys.argv.extend(["--loss_ablation_mode", _mode])
                    print(f">>> [pretrain] loss_ablation_mode = {_mode}")

                # Mở băng N lớp vision cuối để pretrain học đặc trưng thị giác (0/unset = đóng băng).
                _vuf = os.environ.get("VISION_UNFREEZE_LAST_N", "").strip()
                if _vuf:
                    sys.argv.extend(["--vision_unfreeze_last_n", _vuf])
                    print(f">>> [pretrain] vision_unfreeze_last_n = {_vuf}")

                # MLM mask granularity: wholeword (mặc định) | subword (cho ablation A/B).
                _mmm = os.environ.get("MLM_MASK_MODE", "").strip()
                if _mmm:
                    sys.argv.extend(["--mlm_mask_mode", _mmm])
                    print(f">>> [pretrain] mlm_mask_mode = {_mmm}")

                # OCR trong nhánh text của MLM: 0/false = question-only (giảm nạng, mặc định),
                # 1/true = question+OCR (cũ, cho A/B).
                _moit = os.environ.get("MLM_OCR_IN_TEXT", "").strip()
                if _moit:
                    sys.argv.extend(["--mlm_ocr_in_text", _moit])
                    print(f">>> [pretrain] mlm_ocr_in_text = {_moit}")

                # Nếu bật MOCK_TEST, kích hoạt chế độ smoke_test để chạy test nhanh (dataset nhỏ, ít steps)
                if os.environ.get("MOCK_TEST", "").lower() == "true":
                    print("⚠️ [MOCK_TEST] Đang kích hoạt chế độ test nhanh! Sử dụng --smoke_test True.")
                    sys.argv.extend(["--smoke_test", "True"])
                    # Cho phép chỉnh số sample/steps qua env: SMOKE_TRAIN_SAMPLES, SMOKE_EVAL_SAMPLES, SMOKE_MAX_STEPS
                    for _env_key, _flag in [("SMOKE_TRAIN_SAMPLES", "--smoke_train_samples"),
                                            ("SMOKE_EVAL_SAMPLES", "--smoke_eval_samples"),
                                            ("SMOKE_MAX_STEPS", "--smoke_max_steps")]:
                        _v = os.environ.get(_env_key, "").strip()
                        if _v:
                            sys.argv.extend([_flag, _v])
                            print(f"   [MOCK_TEST] {_flag} = {_v}")

                from training import pretrain
                importlib.reload(pretrain)
                pretrain.main()
                print(">>> Pretrain finished successfully.")
            except Exception as e:
                print(f"❌ Error during pretraining: {e}")
                raise e
        elif stage == "predict":
            print("\n>>> [Step 3] Starting Predict (sinh file nộp ID,Answer)...")
            try:
                sys.argv = ["predict.py"]
                #  PREDICT_SPLIT      = dev | test | both (mặc định both)
                #  PREDICT_HF_REPO    = HF repo chứa bundle finetune (model.safetensors +
                #                       config.json + tokenizer). Tự tải về rồi predict.
                #  PREDICT_HF_CKPT    = (tùy chọn) thư mục con checkpoint-XXXX trong repo.
                #  PREDICT_CKPT_DIR   = thư mục bundle finetune LOCAL (ưu tiên hơn HF nếu set).
                #  PREDICT_BATCH_SIZE / PREDICT_NUM_BEAMS = tinh chỉnh generation
                sys.argv.extend(["--split", os.environ.get("PREDICT_SPLIT", "both").strip()])
                _pck = os.environ.get("PREDICT_CKPT_DIR", "").strip()
                _hf_repo = os.environ.get("PREDICT_HF_REPO", "").strip()
                if not _pck and _hf_repo:
                    from huggingface_hub import snapshot_download
                    _sub = os.environ.get("PREDICT_HF_CKPT", "").strip()
                    _dl = os.path.join("./output", "finetune_hf")
                    print(f">>> [predict] Tải bundle finetune từ HF: {_hf_repo}/{_sub or '(root)'}")
                    snapshot_download(
                        repo_id=_hf_repo, repo_type="model", local_dir=_dl,
                        allow_patterns=[f"{_sub}/*"] if _sub else None,
                    )
                    _pck = os.path.join(_dl, _sub) if _sub else _dl
                if _pck:
                    sys.argv.extend(["--ckpt_dir", _pck])
                _pbs = os.environ.get("PREDICT_BATCH_SIZE", "").strip()
                if _pbs:
                    sys.argv.extend(["--batch_size", _pbs])
                _pnb = os.environ.get("PREDICT_NUM_BEAMS", "").strip()
                if _pnb:
                    sys.argv.extend(["--num_beams", _pnb])
                # Kaggle: trỏ vào data attach (tránh gdown Drive)
                for _ek, _eflag in [("PREDICT_IMAGE_DIR", "--image_dir_override"),
                                    ("PREDICT_OCR_DIR", "--ocr_dir_override"),
                                    ("PREDICT_JSON_DIR", "--json_src_dir")]:
                    _ev = os.environ.get(_ek, "").strip()
                    if _ev:
                        sys.argv.extend([_eflag, _ev])
                print(f">>> [predict] argv = {sys.argv}")

                from training import predict
                importlib.reload(predict)
                predict.main()
                print(">>> Predict finished successfully.")
            except Exception as e:
                print(f"❌ Error during predict: {e}")
                raise e
        else:
            print("\n>>> [Step 3] Starting Finetune Training...")
            try:
                sys.argv = ["finetune.py", "configs/finetune.yaml"]

                _fep = os.environ.get("NUM_TRAIN_EPOCHS", "").strip()
                if _fep:
                    sys.argv.extend(["--num_train_epochs", _fep])
                    print(f">>> [finetune] num_train_epochs = {_fep}")

                # Đổi DATASET để finetune (mặc định ViTextVQA trong finetune.yaml):
                #  DATASET_NAME = ViTextVQA | ViOCRVQA | OpenViVQA (phải có configs/data/<ten>.yaml)
                #  DATA_DIR     = thư mục dataset (mặc định ./datasets)
                # Cache CSV đã mang tên dataset (merged_*_<ten>.csv) nên đổi qua lại an toàn.
                for _dk, _dflag in [("DATASET_NAME", "--dataset_name"),
                                    ("DATA_DIR", "--data_dir")]:
                    _dv = os.environ.get(_dk, "").strip()
                    if _dv:
                        sys.argv.extend([_dflag, _dv])
                        print(f">>> [finetune] {_dflag} = {_dv}")

                # Nạp trọng số PRETRAIN để finetune (warm-start):
                #  PRETRAIN_HF_REPO   = HF repo id chứa checkpoint pretrain (vd
                #                       'Kus669/ViSceT5-pretrain-genall'); tự tải về.
                #                       PRETRAIN_HF_CKPT = subfolder (vd 'checkpoint-6591');
                #                       bỏ trống = tự lấy checkpoint-XXXX mới nhất.
                #  MODEL_NAME_OR_PATH = thư mục local chứa model.safetensors của pretrain.
                #  PRETRAIN_WEIGHTS_ID = Drive id zip checkpoint pretrain (tự tải + nạp).
                # Resume một lần finetune đang dở: RESUME_FROM_CHECKPOINT / RESUME_CHECKPOINT_ID.
                _hf_repo = os.environ.get("PRETRAIN_HF_REPO", "").strip()
                if _hf_repo and not os.environ.get("MODEL_NAME_OR_PATH", "").strip():
                    from huggingface_hub import list_repo_files, snapshot_download
                    _sub = os.environ.get("PRETRAIN_HF_CKPT", "").strip()
                    if not _sub:
                        _files = list_repo_files(_hf_repo)
                        _cks = sorted({f.split("/")[0] for f in _files if f.startswith("checkpoint-")},
                                      key=lambda x: int(x.split("-")[1]))
                        _sub = _cks[-1] if _cks else ""
                    _dl = os.path.join("./output/finetune", "pretrain_hf")
                    print(f">>> [finetune] Tải checkpoint pretrain từ HF: {_hf_repo}/{_sub or '(root)'}")
                    snapshot_download(_hf_repo, repo_type="model", local_dir=_dl,
                                      allow_patterns=[f"{_sub}/*"] if _sub else None)
                    os.environ["MODEL_NAME_OR_PATH"] = os.path.join(_dl, _sub) if _sub else _dl

                for _fk, _fflag in [("MODEL_NAME_OR_PATH", "--model_name_or_path"),
                                    ("PRETRAIN_WEIGHTS_ID", "--pretrain_weights_id"),
                                    ("RESUME_FROM_CHECKPOINT", "--resume_from_checkpoint"),
                                    ("RESUME_CHECKPOINT_ID", "--resume_checkpoint_id")]:
                    _fv = os.environ.get(_fk, "").strip()
                    if _fv:
                        sys.argv.extend([_fflag, _fv])
                        print(f">>> [finetune] {_fflag} = {_fv}")

                # Nếu bật MOCK_TEST, kích hoạt chế độ smoke_test để chạy test nhanh (dataset nhỏ, ít steps)
                if os.environ.get("MOCK_TEST", "").lower() == "true":
                    print("⚠️ [MOCK_TEST] Đang kích hoạt chế độ test nhanh! Sử dụng --smoke_test True.")
                    sys.argv.extend(["--smoke_test", "True"])

                from training import finetune
                importlib.reload(finetune)
                finetune.main()
                print(">>> Training finished successfully.")
            except Exception as e:
                print(f"❌ Error during training: {e}")
                raise e

        # 3.5. Tự động upload checkpoint lên Hugging Face Hub (nếu được cấu hình)
        try:
            hf_token = os.environ.get("HF_TOKEN", "")
            hf_repo = os.environ.get("HF_REPO", "")
            
            if hf_token and hf_repo:
                print("\n>>> [Step 3.5] Uploading checkpoints to Hugging Face Hub...")
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                print(f"Creating repository '{hf_repo}' if it doesn't exist...")
                api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
                
                if stage == "pretrain":
                    output_dir = "./output/pretrain"
                elif stage == "predict":
                    output_dir = "./output"   # gồm submission_*.csv để tải về
                else:
                    output_dir = "./output/finetune"
                os.makedirs(output_dir, exist_ok=True)
                # Ghi thông tin meta chạy để thư mục không bao giờ rỗng khi upload test
                with open(os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump({
                        "stage": stage,
                        "mock_test": os.environ.get("MOCK_TEST", "").lower() == "true",
                        "status": "completed"
                    }, f, indent=4)
                
                print(f"Uploading directory '{output_dir}' to '{hf_repo}'...")
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=hf_repo,
                    repo_type="model",
                )
                print(">>> Hugging Face upload successful.")
            else:
                print("\nℹ️ HF_TOKEN or HF_REPO environment variables not found. Skipping Hugging Face upload.")
        except Exception as e:
            print(f"\n❌ Error uploading to Hugging Face: {e}")

    finally:
        # 4. Tự động dừng máy ảo Vast.ai để tránh tốn phí GPU
        try:
            container_label = os.environ.get("VAST_CONTAINERLABEL", "")
            container_api_key = os.environ.get("CONTAINER_API_KEY", "")
            
            if container_label and container_api_key:
                print("\n>>> [Step 4] Auto-stopping Vast.ai instance...")
                # Cài đặt CLI vastai
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vastai"], check=True)
                
                import re
                match = re.search(r"(\d+)(?:\D*)$", container_label)
                instance_id = match.group(1) if match else "".join(re.findall(r"\d+", container_label))
                
                print(f"Stopping instance ID: {instance_id}...")
                subprocess.run([
                    "vastai", "stop", "instance", instance_id, 
                    "--api-key", container_api_key
                ], check=True)
            else:
                print("\nℹ️ Vast.ai environment variables not found. Skipping auto-stop (likely running locally).")
        except Exception as e:
            print(f"\n⚠️ Failed to stop instance automatically: {e}")

if __name__ == "__main__":
    run()
