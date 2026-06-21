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
            from training import finetune
            importlib.reload(finetune)
            finetune.main()
            print(">>> Training finished successfully.")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            raise e

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
