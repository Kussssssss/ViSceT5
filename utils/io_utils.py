import os
import zipfile
import shutil

def _as_id_list(drive_id):
    """Normalize a drive_id into an ordered list of ids (primary first, then
    backups). Accepts a single id, a comma-separated string, or a list."""
    if drive_id is None:
        return []
    if isinstance(drive_id, (list, tuple)):
        items = list(drive_id)
    else:
        items = str(drive_id).split(",")
    return [str(i).strip() for i in items if str(i).strip()]

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
        for idx, fid in enumerate(_as_id_list(drive_id)):
            gdown.download(id=fid, output=output_zip, quiet=False)
            if os.path.exists(output_zip) and os.path.getsize(output_zip) > 0:
                break
            print(f"⚠️  ckpt drive id '{fid}' failed; trying next backup...")

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
    ids = _as_id_list(drive_id)
    if not ids:
        return False

    if os.path.exists(output_path):
        print(f"✅ File already exists at {output_path}. Skipping download.")
        return True

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    for idx, fid in enumerate(ids):
        tag = "primary" if idx == 0 else f"backup #{idx}"
        try:
            gdown.download(id=fid, output=output_path, quiet=False)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                if idx > 0:
                    print(f"✅ Downloaded via {tag} id: {fid}")
                return True
            raise RuntimeError("empty/failed output")
        except Exception as e:
            print(f"⚠️  Drive id '{fid}' ({tag}) failed: {e}")
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except OSError: pass
            if idx + 1 < len(ids):
                print(f"   ↪ trying backup id #{idx + 1}...")
    print(f"❌ All {len(ids)} drive id(s) failed for {output_path}")
    return False
