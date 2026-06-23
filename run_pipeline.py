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

        print("\n>>> [Step 3] Starting Finetune Training...")
        try:
            sys.argv = ["finetune.py", "configs/finetune.yaml"]
            
            # Nếu bật MOCK_TEST, ghi đè các tham số của Hugging Face Trainer để chạy test cực nhanh (2 steps)
            if os.environ.get("MOCK_TEST", "").lower() == "true":
                print("⚠️ [MOCK_TEST] Đang kích hoạt chế độ test nhanh! Đè cấu hình: max_steps=2, logging_steps=1.")
                sys.argv.extend([
                    "--max_steps", "2",
                    "--logging_steps", "1",
                    "--eval_strategy", "no",
                    "--save_strategy", "no"
                ])
                
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
                
                output_dir = "./output/finetune"
                os.makedirs(output_dir, exist_ok=True)
                # Ghi thông tin meta chạy để thư mục không bao giờ rỗng khi upload test
                with open(os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump({
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
