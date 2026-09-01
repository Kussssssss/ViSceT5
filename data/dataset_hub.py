"""
data/dataset_hub.py
DatasetHubLoader — downloads, extracts, builds DataFrames for VQA datasets.
"""

import os, json, zipfile, shutil
from typing import Optional, Dict, Any, List, Tuple, Callable
import pandas as pd

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _is_img(f):
    e = os.path.splitext(f)[1].lower()
    return e in [".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"]

def _is_ocr(f):
    e = os.path.splitext(f)[1].lower()
    return e in [".npy",".npz"]

def _as_id_list(file_id):
    """Primary id first, then backups. Accepts a single id, a comma-separated
    string, or a list — so a config drive_id can carry fallback ids."""
    if file_id is None:
        return []
    items = list(file_id) if isinstance(file_id, (list, tuple)) else str(file_id).split(",")
    return [str(i).strip() for i in items if str(i).strip()]

def _download_gdown_id(file_id: str, out_path: str, fuzzy: bool = True):
    import gdown
    ids = _as_id_list(file_id)
    for idx, fid in enumerate(ids):
        tag = "primary" if idx == 0 else f"backup #{idx}"
        try:
            url = f"https://drive.google.com/uc?id={fid}"
            try:
                gdown.download(id=fid, output=out_path, quiet=False, fuzzy=True)
            except Exception:
                gdown.download(url=url, output=out_path, quiet=False)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                if idx > 0:
                    print(f"✅ Downloaded via {tag} id: {fid}")
                return
            raise RuntimeError("empty/failed output")
        except Exception as e:
            print(f"⚠️  Drive id '{fid}' ({tag}) failed: {e}")
            if os.path.exists(out_path):
                try: os.remove(out_path)
                except OSError: pass
            if idx + 1 < len(ids):
                print(f"   ↪ trying backup id #{idx + 1}...")
    raise RuntimeError(f"All {len(ids)} drive id(s) failed for {out_path}")

def _download_url(url: str, out_path: str):
    import urllib.request
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _maybe_download(src_id: Optional[str], src_url: Optional[str], out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    if src_id:
        _download_gdown_id(src_id, out_path, fuzzy=True)
    elif src_url:
        _download_url(src_url, out_path)
    else:
        raise ValueError("Missing both id and url for download target.")
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Download failed: {out_path}")
    return out_path

def _extract_zip(zip_path: str, dst_dir: str):
    _ensure_dir(dst_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)
    return dst_dir

def _print_tree(root: str, max_files: int = 3):
    for cur, dirs, files in os.walk(root):
        level = cur.replace(root, "").count(os.sep)
        indent = "    " * level
        print(f"{indent}{os.path.basename(cur)}/")
        if len(files) > max_files:
            for f in files[:max_files]:
                print(f"{indent}    {f}")
            print(f"{indent}    ... ({len(files) - max_files} more files)")
        else:
            for f in files:
                print(f"{indent}    {f}")

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _count_in_dir(d, pred):
    try:
        return sum(1 for f in os.listdir(d) if os.path.isfile(os.path.join(d,f)) and pred(f))
    except Exception:
        return 0

def _find_leaf_dir(root: str, pred, max_depth: int = 4) -> str:
    cur = root
    depth = 0
    while depth < max_depth:
        files = [f for f in os.listdir(cur) if os.path.isfile(os.path.join(cur,f))]
        dirs  = [d for d in os.listdir(cur) if os.path.isdir(os.path.join(cur,d))]
        if any(pred(f) for f in files):
            return cur
        if len(dirs) == 1:
            cur = os.path.join(cur, dirs[0]); depth += 1; continue
        if len(dirs) > 1:
            counts = [(d, _count_in_dir(os.path.join(cur,d), pred)) for d in dirs]
            counts.sort(key=lambda x: x[1], reverse=True)
            if counts and counts[0][1] > 0:
                return os.path.join(cur, counts[0][0])
            pref = [d for d in dirs if d.lower() in ["images","imgs","image","img","train","all","data","ocr","ocr_out","ocr_files"]]
            if pref:
                cur = os.path.join(cur, pref[0]); depth += 1; continue
            cur = os.path.join(cur, dirs[0]); depth += 1; continue
        break
    return root

def _safe_rmtree(p):
    try:
        if os.path.isdir(p):
            shutil.rmtree(p)
    except Exception:
        pass

def _safe_remove(p):
    try:
        if os.path.isfile(p):
            os.remove(p)
    except Exception:
        pass

# Index tên-file -> đường dẫn đầy đủ, đệ quy toàn bộ cây thư mục (cache theo root).
# Xử lý bộ dữ liệu tách ảnh/OCR thành NHIỀU thư mục con theo split (OpenViVQA:
# images/{train,dev,test}-images/...), khi _find_leaf_dir chỉ trả về MỘT leaf.
_FILE_INDEX_CACHE: Dict[str, Dict[str, str]] = {}

def _file_index(root, pred):
    if root in _FILE_INDEX_CACHE:
        return _FILE_INDEX_CACHE[root]
    idx = {}
    try:
        for cur, _dirs, files in os.walk(root):
            for f in files:
                if pred(f):
                    idx.setdefault(f, os.path.join(cur, f))
    except Exception:
        pass
    _FILE_INDEX_CACHE[root] = idx
    return idx

def _resolve_in_tree(image_dir, filename, pred):
    """Tìm filename ở bất kỳ thư mục con nào dưới CÙNG cấp với image_dir (parent)."""
    root = os.path.dirname(image_dir.rstrip("/\\")) or image_dir
    return _file_index(root, pred).get(filename)


def _default_vqa_mapper(
    data: Dict[str, Any],
    image_dir: str,
    split_name: str,
    ocr_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    images_dict = {}
    raw_images = data.get("images")
    
    if isinstance(raw_images, list):
        for img in raw_images:
            if isinstance(img, dict):
                images_dict[img.get("id")] = img.get("filename")
    elif isinstance(raw_images, dict):
        for img_id, filename in raw_images.items():
            images_dict[img_id] = filename

    annotations_list = []
    raw_anns = data.get("annotations", [])
    
    if isinstance(raw_anns, list):
        annotations_list = raw_anns
    elif isinstance(raw_anns, dict):
        # Một số bộ (OpenViVQA/VLSP) dùng KHÓA của dict làm question-id còn value
        # KHÔNG có 'id' → list(values()) sẽ mất id submission. Giữ khóa làm 'id'.
        annotations_list = []
        for _k, _v in raw_anns.items():
            if isinstance(_v, dict) and _v.get("id") is None:
                _v = {**_v, "id": int(_k) if str(_k).isdigit() else _k}
            annotations_list.append(_v)

    rows = []
    for ann in annotations_list:
        if not isinstance(ann, dict):
            continue

        img_id = ann.get("image_id")
        
        filename = images_dict.get(img_id)
        if filename is None and img_id is not None:
             filename = images_dict.get(str(img_id))
        
        if filename is None and img_id is not None:
            try:
                filename = images_dict.get(int(img_id))
            except (ValueError, TypeError):
                pass

        question = ann.get("question")
        
        all_answers = None
        if "answers" in ann:
            all_answers = ann.get("answers")
        elif "answer" in ann:
            all_answers = [ann.get("answer")]
        
        answer = None
        if all_answers and isinstance(all_answers, list) and len(all_answers) > 0:
            answer = all_answers[0]

        img_path = os.path.join(image_dir, filename) if filename else None
        # Fallback đa-thư-mục: ảnh không có ở leaf đã chọn → tìm trong toàn cây
        # (vd train image nằm ở train-images/ còn leaf lại là test-images/).
        if img_path and filename and not os.path.isfile(img_path):
            _hit = _resolve_in_tree(image_dir, filename, _is_img)
            if _hit:
                img_path = _hit

        ocr_path = None
        if ocr_dir and filename:
            stem = os.path.splitext(os.path.basename(filename))[0]

            cand_npy = os.path.join(ocr_dir, f"{stem}.npy")
            cand_npz = os.path.join(ocr_dir, f"{stem}.npz")
            if os.path.isfile(cand_npy):
                ocr_path = cand_npy
            elif os.path.isfile(cand_npz):
                ocr_path = cand_npz

            if ocr_path is None and stem.isdigit():
                stem_int = int(stem)
                
                cand_npy_short = os.path.join(ocr_dir, f"{stem_int}.npy")
                cand_npz_short = os.path.join(ocr_dir, f"{stem_int}.npz")
                
                if os.path.isfile(cand_npy_short):
                    ocr_path = cand_npy_short
                elif os.path.isfile(cand_npz_short):
                    ocr_path = cand_npz_short

            if ocr_path is None and stem.isdigit():
                 stem_12 = f"{int(stem):012d}"
                 cand_npy_12 = os.path.join(ocr_dir, f"{stem_12}.npy")
                 cand_npz_12 = os.path.join(ocr_dir, f"{stem_12}.npz")
                 if os.path.isfile(cand_npy_12):
                     ocr_path = cand_npy_12
                 elif os.path.isfile(cand_npz_12):
                     ocr_path = cand_npz_12

            # Fallback đa-thư-mục cho OCR (train-ocr/, dev-ocr/, ...): tìm theo tên
            if ocr_path is None:
                for _cand in (f"{stem}.npy", f"{stem}.npz",
                              *(( f"{int(stem)}.npy", f"{int(stem)}.npz",
                                  f"{int(stem):012d}.npy", f"{int(stem):012d}.npz")
                                if stem.isdigit() else ())):
                    _hit = _resolve_in_tree(ocr_dir, _cand, _is_ocr)
                    if _hit:
                        ocr_path = _hit; break

        rows.append({
            "dataset": None,
            "split": split_name,
            "id": ann.get("id"),          # annotation/question id — submission ID (ID,Answer)
            "image_id": img_id,
            "image_filename": filename,
            "question": question,
            "answer": answer,
            "all_answers": all_answers,
            "image_path": img_path,
            "ocr_path": ocr_path,
        })
    return rows

class DatasetHubLoader:
    def __init__(self, raw_root: str, out_root: str, cleanup: bool = True):
        self.raw_root = _ensure_dir(raw_root)
        self.out_root = _ensure_dir(out_root)
        self.cleanup = cleanup
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.paths: Dict[str, Dict[str, Any]] = {}

    def register_dataset(
        self,
        dataset_name: str,
        task_type: str,
        image_zip_id: Optional[str] = None,
        image_dir_override: Optional[str] = None,
        image_zip_name: str = "images.zip",
        image_subdir_name: str = "images",
        ocr_zip_id: Optional[str] = None,
        ocr_dir_override: Optional[str] = None,
        ocr_zip_name: str = "ocr.zip",
        ocr_subdir_name: str = "ocr",
        splits: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        mapper_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        image_zip_path: Optional[str] = None,
        ocr_zip_path: Optional[str] = None,
    ):
        if dataset_name in self.registry:
            raise ValueError("Dataset already registered")
        self.registry[dataset_name] = {
            "task_type": task_type,
            "image_zip_id": image_zip_id,
            "image_dir_override": image_dir_override,
            "image_zip_name": image_zip_name,
            "image_subdir_name": image_subdir_name,
            "ocr_zip_id": ocr_zip_id,
            "ocr_dir_override": ocr_dir_override,
            "ocr_zip_name": ocr_zip_name,
            "ocr_subdir_name": ocr_subdir_name,
            "splits": splits or {},
            "mapper_fn": mapper_fn or _default_vqa_mapper,
            "image_zip_path": image_zip_path,
            "ocr_zip_path": ocr_zip_path,
        }

    def prepare(self, dataset_name: str) -> Dict[str, Any]:
        if dataset_name not in self.registry:
            raise KeyError("Dataset not registered")
        spec = self.registry[dataset_name]
        out_dir = _ensure_dir(os.path.join(self.out_root, dataset_name))
        ann_out = {}
        has_any_split = any((conf.get("id") and str(conf.get("id")).strip()) or conf.get("url") for conf in spec["splits"].values())
        if not has_any_split:
            print(f"ℹ️ [Hub] Dataset '{dataset_name}' là tập Image+OCR thuần túy (không có file QA JSON).")
        else:
            for split, conf in spec["splits"].items():
                id_ = conf.get("id")
                url = conf.get("url")
                if not (id_ and str(id_).strip()) and not url:
                    continue
                out_p = os.path.join(out_dir, f"{dataset_name}_{split}.json")
                _maybe_download(id_, url, out_p)
                ann_out[split] = out_p

        image_dir, ocr_dir = None, None

        if spec["image_dir_override"] and os.path.isdir(spec["image_dir_override"]):
            image_dir = _find_leaf_dir(spec["image_dir_override"], _is_img)
        else:
            image_root = os.path.join(out_dir, spec["image_subdir_name"])
            
            if os.path.isdir(image_root) and _count_in_dir(_find_leaf_dir(image_root, _is_img), _is_img) > 0:
                image_dir = _find_leaf_dir(image_root, _is_img)
            else:
                raw_dir_img = _ensure_dir(os.path.join(self.raw_root, f"{dataset_name}_img"))
                zip_path_img = spec.get("image_zip_path")

                if zip_path_img is None:
                    if not spec["image_zip_id"]:
                        raise ValueError("image_zip_id or image_zip_path is required when image_dir_override is not provided.")
                    zip_path_img = os.path.join(raw_dir_img, spec["image_zip_name"])
                    print(f"⬇️ [Hub] Đang tải tập ảnh '{dataset_name}' từ Google Drive (ID: {spec['image_zip_id']})...")
                    _maybe_download(spec["image_zip_id"], None, zip_path_img)

                print(f"📦 [Hub] Đang giải nén tập ảnh '{dataset_name}'...")
                _extract_zip(zip_path_img, image_root)
                image_dir = _find_leaf_dir(image_root, _is_img)

                if self.cleanup and spec.get("image_zip_path") is None:
                    _safe_remove(zip_path_img)
                    _safe_rmtree(raw_dir_img)

        if spec["ocr_dir_override"] and os.path.isdir(spec["ocr_dir_override"]):
            ocr_dir = _find_leaf_dir(spec["ocr_dir_override"], _is_ocr)
        else:
            ocr_root = os.path.join(out_dir, spec["ocr_subdir_name"])
            
            if os.path.isdir(ocr_root) and _count_in_dir(_find_leaf_dir(ocr_root, _is_ocr), _is_ocr) > 0:
                ocr_dir = _find_leaf_dir(ocr_root, _is_ocr)
            else:
                if spec.get("ocr_zip_path") or spec.get("ocr_zip_id"):
                    raw_dir_ocr = _ensure_dir(os.path.join(self.raw_root, f"{dataset_name}_ocr"))
                    zip_path_ocr = spec.get("ocr_zip_path")
                    if zip_path_ocr is None:
                        zip_path_ocr = os.path.join(raw_dir_ocr, spec["ocr_zip_name"])
                        print(f"⬇️ [Hub] Đang tải tập OCR '{dataset_name}' từ Google Drive (ID: {spec['ocr_zip_id']})...")
                        _maybe_download(spec["ocr_zip_id"], None, zip_path_ocr)

                    print(f"📦 [Hub] Đang giải nén tập OCR '{dataset_name}'...")
                    _extract_zip(zip_path_ocr, ocr_root)
                    ocr_dir = _find_leaf_dir(ocr_root, _is_ocr)

                    if self.cleanup and spec.get("ocr_zip_path") is None:
                        _safe_remove(zip_path_ocr)
                        _safe_rmtree(raw_dir_ocr)

        self.paths[dataset_name] = {
            "out_dir": out_dir,
            "annotations": ann_out,
            "image_dir": image_dir,
            "ocr_dir": ocr_dir,
            "image_root": image_root,
            "ocr_root": ocr_root,
        }
        return self.paths[dataset_name]

    def build_df(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name not in self.paths:
            raise RuntimeError("Call prepare() first")
        spec = self.registry[dataset_name]
        p = self.paths[dataset_name]
        ann = p["annotations"]
        img_dir = p["image_dir"]
        ocr_dir = p.get("ocr_dir")
        img_root = p.get("image_root") or img_dir
        ocr_root = p.get("ocr_root") or ocr_dir
        mapper = spec["mapper_fn"]
        rows = []
        if ann:
            for split, json_path in ann.items():
                # Bỏ qua split có JSON hỏng (vd file tải nhầm không phải JSON) thay vì
                # crash cả build — split khác vẫn dùng được (finetune chỉ cần train/val).
                try:
                    data = _load_json(json_path)
                except Exception as e:
                    print(f"⚠️ [Hub] Bỏ qua split '{split}': đọc JSON lỗi ({type(e).__name__}: {e}).")
                    continue
                rows.extend(mapper(data, img_dir, split, ocr_dir))
        elif img_root and os.path.isdir(img_root):
            # PURE IMAGE + OCR PAIRS (e.g. VinText, EVJVQA pre-training datasets without separate QA json)
            # Scan toàn bộ cây thư mục từ img_root và ocr_root để không bỏ sót các thư mục con (train, val, test)
            img_index = _file_index(img_root, _is_img)
            ocr_index = _file_index(ocr_root, _is_ocr) if (ocr_root and os.path.isdir(ocr_root)) else {}
            
            items = []
            for fname, fpath in img_index.items():
                stem = os.path.splitext(fname)[0]
                ocr_path = None
                for ext in [".npy", ".npz"]:
                    if f"{stem}{ext}" in ocr_index:
                        ocr_path = ocr_index[f"{stem}{ext}"]
                        break
                if ocr_path is None and stem.isdigit():
                    stem_int = str(int(stem))
                    for ext in [".npy", ".npz"]:
                        if f"{stem_int}{ext}" in ocr_index:
                            ocr_path = ocr_index[f"{stem_int}{ext}"]
                            break
                if ocr_path is None and stem.isdigit():
                    stem_12 = f"{int(stem):012d}"
                    for ext in [".npy", ".npz"]:
                        if f"{stem_12}{ext}" in ocr_index:
                            ocr_path = ocr_index[f"{stem_12}{ext}"]
                            break
                items.append({
                    "image_filename": fname,
                    "image_path": fpath,
                    "ocr_path": ocr_path,
                })
            
            items.sort(key=lambda x: x["image_filename"])
            n_val = max(1, int(len(items) * 0.05)) if len(items) > 20 else max(1, len(items) // 5)
            
            for idx, it in enumerate(items):
                split = "validation" if idx >= len(items) - n_val else "train"
                rows.append({
                    "dataset": None,
                    "split": split,
                    "id": f"{dataset_name}_{idx}",
                    "image_id": idx,
                    "image_filename": it["image_filename"],
                    "question": "",
                    "answer": "",
                    "all_answers": [],
                    "image_path": it["image_path"],
                    "ocr_path": it["ocr_path"],
                })
        df = pd.DataFrame(rows)
        if "dataset" in df.columns:
            df["dataset"] = dataset_name
        else:
            df.insert(0, "dataset", dataset_name)
        return df

    def load_task(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        df = self.build_df(dataset_name)
        return {
            "train": df[df["split"].str.lower().isin(["train", "training"])].reset_index(drop=True),
            "validation": df[df["split"].str.lower().isin(["val", "valid", "validation", "dev"])].reset_index(drop=True),
            "test": df[df["split"].str.lower().isin(["test", "testing"])].reset_index(drop=True),
        }

    def stats(self, dataset_name: str, df_all: pd.DataFrame):
        p = self.paths.get(dataset_name, {})
        print(f"Dataset: {dataset_name}")
        print(f"Total: {len(df_all):,}")
        for s in ["train","validation","test"]:
            print(f"{s.capitalize()}: {len(df_all[df_all['split'].str.lower().isin([s, 'valid' if s=='validation' else s, 'dev' if s=='validation' else s])]):,}")
        img_dir = p.get("image_dir")
        ocr_dir = p.get("ocr_dir")
        if img_dir and os.path.isdir(img_dir):
            try:
                unique_in_df = df_all['image_filename'].nunique()
                files_in_dir = len([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and _is_img(f)])
                print(f"Image files in {img_dir}: {files_in_dir:,} (unique filenames in DF: {unique_in_df:,})")
            except Exception as e:
                print(f"Image stats error: {e}")
        if ocr_dir and os.path.isdir(ocr_dir):
            try:
                ocr_files = len([f for f in os.listdir(ocr_dir) if os.path.isfile(os.path.join(ocr_dir, f)) and _is_ocr(f)])
                print(f"OCR files in {ocr_dir}: {ocr_files:,}")
            except Exception as e:
                print(f"OCR stats error: {e}")

    def show_layout(self, dataset_name: str):
        p = self.paths.get(dataset_name, {})
        out_dir = p.get("out_dir", os.path.join(self.out_root, dataset_name))
        print("OUT:")
        _print_tree(out_dir)