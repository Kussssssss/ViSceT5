import os
import zipfile
import shutil

def unzip_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_to)

def download_and_extract_checkpoint(drive_id, extract_to):
    import gdown
    if not drive_id: 
        return False
        
    # Check if we already have something in extract_to (assuming config.json means it's downloaded)
    if os.path.exists(extract_to) and os.path.exists(os.path.join(extract_to, "config.json")):
        print(f"✅ Checkpoint directory already exists and seems valid at {extract_to}. Skipping download.")
        return True

    try:
        output_zip = os.path.join(os.path.dirname(extract_to), "downloaded_ckpt.zip")
        os.makedirs(os.path.dirname(extract_to), exist_ok=True)
        gdown.download(id=drive_id, output=output_zip, quiet=False)
        
        if os.path.exists(output_zip):
            os.makedirs(extract_to, exist_ok=True)
            unzip_folder(output_zip, extract_to)
            os.remove(output_zip)
            
            # Flatten if nested
            contents = os.listdir(extract_to)
            subdirs = [d for d in contents if os.path.isdir(os.path.join(extract_to, d))]
            if len(subdirs) == 1 and "config.json" not in contents:
                nested_dir = os.path.join(extract_to, subdirs[0])
                for f in os.listdir(nested_dir):
                    shutil.move(os.path.join(nested_dir, f), extract_to)
                os.rmdir(nested_dir)
            return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
    return False

def download_file(drive_id, output_path):
    import gdown
    if not drive_id: 
        return False
        
    if os.path.exists(output_path):
        print(f"✅ File already exists at {output_path}. Skipping download.")
        return True

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gdown.download(id=drive_id, output=output_path, quiet=False)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
    return False
