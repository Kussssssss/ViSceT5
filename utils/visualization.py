"""
utils/visualization.py
Visualization helpers: sample display, OCR boxes, heatmaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import json

sample_id = 0

IMAGE_DIR = "/kaggle/input/vitextvqa-viocrvqa/ViTextVQA_images/st_images"
OCR_DIR = "/kaggle/input/vitextvqa-viocrvqa/OCR_ViTextVQA/swintextspotter"
annotation_path = f"/kaggle/working/datasets/{NAME_SET1}/{NAME_SET1}_train.json"


def build_id_variants(target_id):
    raw_id = str(target_id)
    padded_id = raw_id.zfill(12)

    return {
        "raw_id": raw_id,
        "padded_id": padded_id,
        "raw_image": f"{raw_id}.jpg",
        "padded_image": f"{padded_id}.jpg",
        "raw_ocr": f"{raw_id}.npy",
        "padded_ocr": f"{padded_id}.npy",
    }


def find_existing_file(base_dir, candidates):
    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path

    return os.path.join(base_dir, candidates[0])


def resolve_paths(target_id):
    ids = build_id_variants(target_id)

    img_path = find_existing_file(
        IMAGE_DIR,
        [ids["raw_image"], ids["padded_image"]]
    )

    ocr_path = find_existing_file(
        OCR_DIR,
        [ids["raw_ocr"], ids["padded_ocr"]]
    )

    return img_path, ocr_path, ids


def normalize_filename(value):
    if not value:
        return ""
    return os.path.basename(str(value))


def ensure_list(obj):
    """
    Chuẩn hóa dữ liệu về list.
    Hỗ trợ:
    - list
    - dict có key số/string
    - dict dạng {"0": {...}, "1": {...}}
    - dict dạng {"abc": {...}}
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        return list(obj.values())

    return []


def extract_annotation_list(data):
    """
    Hỗ trợ nhiều cấu trúc JSON:
    1. List trực tiếp:
       [{...}, {...}]

    2. Dict có data:
       {"data": [...]}

    3. Dict có annotations:
       {"annotations": [...]}

    4. COCO-style:
       {"images": [...], "annotations": [...]}

    5. Dict annotations là dict:
       {"annotations": {"0": {...}, "1": {...}}}
    """
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    keys = list(data.keys())
    print(f"🔍 JSON Keys detected: {keys}")

    if "annotations" in data:
        return ensure_list(data["annotations"])

    if "data" in data:
        return ensure_list(data["data"])

    if "dataset" in data:
        return ensure_list(data["dataset"])

    if "question" in data and ("answer" in data or "answers" in data):
        return [data]

    return ensure_list(data)


def extract_images_info(data):
    """
    Nếu JSON có key images thì lấy ra để map image_id -> file_name.
    """
    if not isinstance(data, dict):
        return {}

    images = ensure_list(data.get("images", []))
    image_map = {}

    for img in images:
        if not isinstance(img, dict):
            continue

        img_id = str(img.get("id", img.get("image_id", "")))
        file_name = normalize_filename(
            img.get("file_name", img.get("image", ""))
        )

        if img_id:
            image_map[img_id] = file_name

    return image_map


def is_match_annotation(item, ids, possible_filenames, image_map=None):
    if not isinstance(item, dict):
        return False

    image_map = image_map or {}

    possible_ids = {
        ids["raw_id"],
        ids["padded_id"],
    }

    if ids["raw_id"].isdigit():
        possible_ids.add(str(int(ids["raw_id"])))

    item_img_id = str(
        item.get(
            "image_id",
            item.get("img_id", item.get("id", ""))
        )
    )

    item_filename = normalize_filename(
        item.get(
            "image",
            item.get("file_name", item.get("filename", ""))
        )
    )

    mapped_filename = normalize_filename(image_map.get(item_img_id, ""))
    item_filename_no_ext = os.path.splitext(item_filename)[0]
    mapped_filename_no_ext = os.path.splitext(mapped_filename)[0]

    matched_by_id = item_img_id in possible_ids
    matched_by_filename = item_filename in possible_filenames
    matched_by_mapped_filename = mapped_filename in possible_filenames
    matched_by_filename_id = item_filename_no_ext in possible_ids
    matched_by_mapped_filename_id = mapped_filename_no_ext in possible_ids

    return (
        matched_by_id
        or matched_by_filename
        or matched_by_mapped_filename
        or matched_by_filename_id
        or matched_by_mapped_filename_id
    )


def visualize_data(target_id, annotation_path):
    img_path, ocr_path, ids = resolve_paths(target_id)

    print("=" * 60)
    print(f"🚀 PROCESSING SAMPLE ID: {target_id}")
    print(f"🔎 Try IDs: {ids['raw_id']} / {ids['padded_id']}")
    print(f"🖼️ Image path: {img_path}")
    print(f"📂 OCR path: {ocr_path}")
    print("=" * 60)

    found_qa = []

    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"📚 Annotation file loaded. Type: {type(data)}")

            data_list = extract_annotation_list(data)
            image_map = extract_images_info(data)

            if len(data_list) > 0:
                first_item = data_list[0]
                if isinstance(first_item, str):
                    print("❌ Error: Data list contains strings, expected dictionaries.")
                    data_list = []

            possible_filenames = {
                ids["raw_image"],
                ids["padded_image"],
                os.path.basename(img_path),
            }

            for item in data_list:
                if is_match_annotation(item, ids, possible_filenames, image_map):
                    found_qa.append(item)

            if found_qa:
                print(f"\n❓ Found {len(found_qa)} Question(s):")
                for idx, qa in enumerate(found_qa):
                    q_text = qa.get("question", qa.get("query", "N/A"))
                    a_text = qa.get("answers", qa.get("answer", qa.get("label", "N/A")))

                    print(f"  🔹 Pair {idx + 1}:")
                    print(f"     Q: {q_text}")
                    print(f"     A: {a_text}")
            else:
                print(
                    f"⚠️ No QA found for ID {ids['raw_id']} / "
                    f"{ids['padded_id']} or filenames {possible_filenames}"
                )

        except Exception as e:
            print(f"❌ Error reading annotation file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Annotation file not found at: {annotation_path}")

    print("-" * 60)

    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    print(f"🖼️ Image Loaded: {img_path} ({img_w}x{img_h})")

    if not os.path.exists(ocr_path):
        print(f"⚠️ OCR not found: {ocr_path}")
        return

    ocr_data = np.load(ocr_path, allow_pickle=True)

    if isinstance(ocr_data, np.ndarray) and ocr_data.ndim == 0:
        ocr_data = ocr_data.item()

    print("📂 OCR Loaded.")

    if isinstance(ocr_data, dict):
        boxes = ocr_data.get("boxes", [])
        texts = ocr_data.get("texts", [])

        if len(texts) == 0:
            texts = ocr_data.get("ocr_tokens", [])
    else:
        boxes = []
        texts = []

    print(f"📊 Found {len(boxes)} OCR text boxes.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    for i, box in enumerate(boxes):
        text = texts[i] if i < len(texts) else ""

        box = np.array(box).astype(float)

        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        if np.max(box) <= 1.5:
            x1 *= img_w
            x2 *= img_w
            y1 *= img_h
            y2 *= img_h

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1 - 5,
            str(text),
            color="black",
            fontsize=8,
            weight="bold",
            bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="none"),
        )

    plt.axis("off")
    plt.show()


visualize_data(sample_id, annotation_path)

import os
from PIL import Image, ExifTags
from tqdm import tqdm

def check_rotated_images(image_dir):
    rotated_count = 0
    total_count = 0
    rotated_files = []

    ORIENTATION_TAG = 274

    print(f"📂 Scanning directory: {image_dir}")
    
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files):
        img_path = os.path.join(image_dir, filename)
        try:
            with Image.open(img_path) as img:
                exif = img.getexif()
                if exif and ORIENTATION_TAG in exif:
                    orientation = exif[ORIENTATION_TAG]
                    if orientation != 1:
                        rotated_count += 1
                        rotated_files.append((filename, orientation))
                total_count += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print("="*40)
    print(f"📊 REPORT:")
    print(f"Total images scanned: {total_count}")
    print(f"Images requiring rotation (EXIF != 1): {rotated_count}")
    print(f"Percentage: {rotated_count/total_count*100:.2f}%")
    print("="*40)
    
    if rotated_files:
        print("Example rotated files (filename, orientation_code):")
        print(rotated_files[:5]) # In thử 5 file đầu tiên


check_rotated_images(IMAGE_DIR)

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

SAMPLE_ID_TO_VISUALIZE = 0

IMAGE_DIR = "/kaggle/input/images/images"
OCR_DIR = "/kaggle/input/vitextvqa-ocr-extracting/swintextspotter"
ANNOTATION_PATH = f"/kaggle/working/datasets/{NAME_SET1}/{NAME_SET1}_train.json"


# ============================================================
# ID / PATH HELPERS
# ============================================================

def build_id_variants(target_id):
    raw_id = str(target_id)
    padded_12_id = raw_id.zfill(12)

    return {
        "raw_id": raw_id,
        "padded_12_id": padded_12_id,

        "raw_jpg": f"{raw_id}.jpg",
        "padded_12_jpg": f"{padded_12_id}.jpg",

        "raw_jpeg": f"{raw_id}.jpeg",
        "padded_12_jpeg": f"{padded_12_id}.jpeg",

        "raw_png": f"{raw_id}.png",
        "padded_12_png": f"{padded_12_id}.png",

        "raw_npy": f"{raw_id}.npy",
        "padded_12_npy": f"{padded_12_id}.npy",
    }


def find_existing_file(base_dir, candidates):
    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            return path

    return os.path.join(base_dir, candidates[0])


def resolve_sample_paths(target_id, image_dir, ocr_dir):
    ids = build_id_variants(target_id)

    img_path = find_existing_file(
        image_dir,
        [
            ids["raw_jpg"],
            ids["padded_12_jpg"],
            ids["raw_jpeg"],
            ids["padded_12_jpeg"],
            ids["raw_png"],
            ids["padded_12_png"],
        ]
    )

    ocr_path = find_existing_file(
        ocr_dir,
        [
            ids["raw_npy"],
            ids["padded_12_npy"],
        ]
    )

    return img_path, ocr_path, ids


# ============================================================
# SAFE UTILS
# ============================================================

def safe_len(value):
    try:
        return len(value)
    except Exception:
        return 0


def is_non_empty(value):
    if value is None:
        return False

    if isinstance(value, np.ndarray):
        return value.size > 0

    try:
        return len(value) > 0
    except Exception:
        return True


def get_first_existing_key(data, keys, default=None):
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def normalize_filename(value):
    if not value:
        return ""
    return os.path.basename(str(value))


def ensure_list(obj):
    if obj is None:
        return []

    if isinstance(obj, list):
        return obj

    if isinstance(obj, tuple):
        return list(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return list(obj.values())

    return []


# ============================================================
# ANNOTATION HELPERS
# ============================================================

def extract_annotation_list(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    keys = list(data.keys())
    print(f"🔍 JSON Keys detected: {keys}")

    if "annotations" in data:
        return ensure_list(data["annotations"])

    if "data" in data:
        return ensure_list(data["data"])

    if "dataset" in data:
        return ensure_list(data["dataset"])

    if "question" in data and ("answer" in data or "answers" in data):
        return [data]

    return ensure_list(data)


def extract_images_info(data):
    if not isinstance(data, dict):
        return {}

    images = ensure_list(data.get("images", []))
    image_map = {}

    for img in images:
        if not isinstance(img, dict):
            continue

        img_id = str(img.get("id", img.get("image_id", "")))
        file_name = normalize_filename(
            img.get("file_name", img.get("image", img.get("filename", "")))
        )

        if img_id:
            image_map[img_id] = file_name

    return image_map


def is_match_annotation(item, ids, possible_filenames, image_map=None):
    if not isinstance(item, dict):
        return False

    image_map = image_map or {}

    possible_ids = {
        ids["raw_id"],
        ids["padded_12_id"],
    }

    if ids["raw_id"].isdigit():
        possible_ids.add(str(int(ids["raw_id"])))

    item_img_id = str(
        item.get(
            "image_id",
            item.get("img_id", item.get("id", ""))
        )
    )

    item_filename = normalize_filename(
        item.get(
            "image",
            item.get("file_name", item.get("filename", ""))
        )
    )

    mapped_filename = normalize_filename(image_map.get(item_img_id, ""))

    item_filename_no_ext = os.path.splitext(item_filename)[0]
    mapped_filename_no_ext = os.path.splitext(mapped_filename)[0]

    return (
        item_img_id in possible_ids
        or item_filename in possible_filenames
        or mapped_filename in possible_filenames
        or item_filename_no_ext in possible_ids
        or mapped_filename_no_ext in possible_ids
    )


def load_annotations_for_id(annotation_path, ids, img_path):
    found_qa = []

    if not annotation_path or not os.path.exists(annotation_path):
        print(f"⚠️ Annotation file not found: {annotation_path}")
        return found_qa

    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"📚 Annotation file loaded. Type: {type(data)}")

        data_list = extract_annotation_list(data)
        image_map = extract_images_info(data)

        possible_filenames = {
            ids["raw_jpg"],
            ids["padded_12_jpg"],
            ids["raw_jpeg"],
            ids["padded_12_jpeg"],
            ids["raw_png"],
            ids["padded_12_png"],
            os.path.basename(img_path),
        }

        for item in data_list:
            if is_match_annotation(item, ids, possible_filenames, image_map):
                found_qa.append(item)

        if found_qa:
            print(f"\n❓ Found {len(found_qa)} Question(s):")
            for idx, qa in enumerate(found_qa):
                q_text = qa.get("question", qa.get("query", "N/A"))
                a_text = qa.get(
                    "answers",
                    qa.get("answer", qa.get("label", qa.get("gt_answer", "N/A")))
                )

                print(f"  🔹 Pair {idx + 1}:")
                print(f"     Q: {q_text}")
                print(f"     A: {a_text}")
        else:
            print("⚠️ No QA found for this sample.")

    except Exception as e:
        print(f"❌ Error reading annotation file: {e}")
        import traceback
        traceback.print_exc()

    return found_qa


# ============================================================
# OCR HELPERS
# ============================================================

def load_npy_safely(npy_path):
    data = np.load(npy_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.ndim == 0:
        try:
            data = data.item()
        except Exception:
            pass

    return data


def extract_ocr_boxes_texts(ocr_data):
    boxes, texts = [], []

    if isinstance(ocr_data, dict):
        boxes = get_first_existing_key(
            ocr_data,
            [
                "boxes",
                "bbox",
                "bboxes",
                "ocr_boxes",
                "det_boxes",
                "polys",
                "polygons",
                "points",
            ],
            default=[]
        )

        texts = get_first_existing_key(
            ocr_data,
            [
                "texts",
                "text",
                "ocr_tokens",
                "tokens",
                "words",
                "rec_texts",
                "transcriptions",
            ],
            default=[]
        )

    elif isinstance(ocr_data, np.ndarray):
        if ocr_data.dtype == object:
            ocr_list = ocr_data.tolist()

            if isinstance(ocr_list, dict):
                return extract_ocr_boxes_texts(ocr_list)

            if isinstance(ocr_list, list):
                if len(ocr_list) > 0 and isinstance(ocr_list[0], dict):
                    for item in ocr_list:
                        box = get_first_existing_key(
                            item,
                            ["box", "bbox", "boxes", "points", "polygon", "poly"],
                            default=None
                        )

                        text = get_first_existing_key(
                            item,
                            ["text", "word", "token", "transcription"],
                            default=""
                        )

                        if box is not None:
                            boxes.append(box)
                            texts.append(text)
                else:
                    boxes = ocr_list
                    texts = [""] * len(boxes)

        else:
            if ocr_data.ndim == 2 and ocr_data.shape[1] >= 4:
                boxes = ocr_data[:, :4].tolist()
                texts = [""] * len(boxes)

    elif isinstance(ocr_data, list):
        if len(ocr_data) > 0 and isinstance(ocr_data[0], dict):
            for item in ocr_data:
                box = get_first_existing_key(
                    item,
                    ["box", "bbox", "boxes", "points", "polygon", "poly"],
                    default=None
                )

                text = get_first_existing_key(
                    item,
                    ["text", "word", "token", "transcription"],
                    default=""
                )

                if box is not None:
                    boxes.append(box)
                    texts.append(text)
        else:
            boxes = ocr_data
            texts = [""] * len(boxes)

    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    if not is_non_empty(texts):
        texts = [""] * safe_len(boxes)

    return boxes, texts


def normalize_box_to_xyxy(box, img_w, img_h):
    try:
        box = np.array(box).astype(float).reshape(-1)
    except Exception:
        return None

    if len(box) < 4:
        return None

    if len(box) >= 8:
        xs = box[0::2]
        ys = box[1::2]

        x1, x2 = np.nanmin(xs), np.nanmax(xs)
        y1, y2 = np.nanmin(ys), np.nanmax(ys)
    else:
        x1, y1, x2, y2 = box[:4]

    if np.nanmax(box) <= 1.5:
        x1 *= img_w
        x2 *= img_w
        y1 *= img_h
        y2 *= img_h

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


# ============================================================
# ROTATION VERIFY
# ============================================================

def verify_rotation_fix(image_dir, num_samples=5, max_scan=None):
    print("=" * 60)
    print("🧪 VERIFYING ROTATION FIX (ImageOps.exif_transpose)")
    print("=" * 60)

    if not os.path.exists(image_dir):
        print(f"❌ Image dir not found: {image_dir}")
        return

    files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    ORIENTATION_TAG = 274
    count = 0
    scanned = 0

    for filename in tqdm(files, desc="Searching for rotated images"):
        if count >= num_samples:
            break

        if max_scan is not None and scanned >= max_scan:
            break

        scanned += 1
        img_path = os.path.join(image_dir, filename)

        try:
            with Image.open(img_path) as img:
                exif = img.getexif()
                orientation = exif.get(ORIENTATION_TAG, 1) if exif else 1

                if orientation != 1:
                    print(f"\n📸 Found Rotated Image: {filename}")
                    print(f"   👉 Orientation Code: {orientation}")
                    print(f"   👉 Original Size Raw: {img.size}")

                    fixed_img = ImageOps.exif_transpose(img)

                    print(f"   👉 Fixed Size Transposed: {fixed_img.size}")

                    if img.size != fixed_img.size and orientation in [5, 6, 7, 8]:
                        print("   ✅ SUCCESS: Dimensions swapped correctly.")
                    elif orientation in [3, 4]:
                        print("   ✅ SUCCESS: Dimensions same, pixels reordered.")
                    else:
                        print("   ❓ Orientation present but dimensions unchanged.")

                    count += 1

        except Exception:
            continue

    if count == 0:
        print("\n⚠️ No rotated images found in scanned images.")


# ============================================================
# VISUALIZE SAMPLE
# ============================================================

def visualize_sample_fixed(
    target_id,
    image_dir,
    ocr_dir,
    annotation_path=None,
    show_annotations=True,
    figsize=(12, 12)
):
    img_path, ocr_path, ids = resolve_sample_paths(target_id, image_dir, ocr_dir)

    print("\n" + "=" * 60)
    print(f"🚀 VISUALIZING SAMPLE ID: {target_id}")
    print(f"🔎 Try IDs: {ids['raw_id']} / {ids['padded_12_id']}")
    print(f"🖼️ Image path: {img_path}")
    print(f"📂 OCR path: {ocr_path}")
    print("=" * 60)

    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    if show_annotations and annotation_path:
        load_annotations_for_id(annotation_path, ids, img_path)
        print("-" * 60)

    raw_image = Image.open(img_path)
    raw_exif = raw_image.getexif()
    orientation = raw_exif.get(274, 1) if raw_exif else 1

    print(f"📷 Original EXIF Orientation: {orientation}")

    fixed_image = ImageOps.exif_transpose(raw_image).convert("RGB")
    img_w, img_h = fixed_image.size

    print(f"🖼️ Processed Image Size: {img_w}x{img_h}")

    if os.path.exists(ocr_path):
        try:
            ocr_data = load_npy_safely(ocr_path)
            boxes, texts = extract_ocr_boxes_texts(ocr_data)
            print(f"📂 OCR Loaded. Found {len(boxes)} text boxes.")
        except Exception as e:
            print(f"❌ Error loading OCR: {e}")
            boxes, texts = [], []
    else:
        print(f"⚠️ OCR file not found: {ocr_path}")
        boxes, texts = [], []

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(fixed_image)

    valid_box_count = 0

    for i, box in enumerate(boxes):
        xyxy = normalize_box_to_xyxy(box, img_w, img_h)

        if xyxy is None:
            continue

        x1, y1, x2, y2 = xyxy
        text = texts[i] if i < len(texts) else ""

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        if text:
            ax.text(
                x1,
                max(y1 - 5, 0),
                str(text),
                color="black",
                fontsize=9,
                weight="bold",
                bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="none")
            )

        valid_box_count += 1

    print(f"✅ Visualized {valid_box_count}/{len(boxes)} valid OCR boxes.")

    plt.title(
        f"Corrected Image ID: {target_id} | File: {os.path.basename(img_path)}",
        fontsize=14
    )

    plt.axis("off")
    plt.show()


# ============================================================
# RUN
# ============================================================

verify_rotation_fix(
    image_dir=IMAGE_DIR,
    num_samples=3,
    max_scan=None
)

visualize_sample_fixed(
    target_id=SAMPLE_ID_TO_VISUALIZE,
    image_dir=IMAGE_DIR,
    ocr_dir=OCR_DIR,
    annotation_path=ANNOTATION_PATH,
    show_annotations=True,
    figsize=(12, 12)
)

import os, numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

vision_ocr = Vision_Encode_Ocr_Feature(ocr_config)

def _draw_boxes_in_order(image_path, boxes_norm, title=""):
    if not os.path.exists(image_path):
        print("❌ Không tìm thấy ảnh:", image_path)
        return

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Lỗi mở ảnh {image_path}: {e}")
        return

    W, H = img.size

    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_title(f"{title}\n({W}x{H})", fontsize=10)

    if boxes_norm is not None and isinstance(boxes_norm, torch.Tensor) and boxes_norm.numel() > 0:
        b = boxes_norm.clone().float()
        b_px = b * torch.tensor([W, H, W, H], dtype=torch.float32)

        for idx in range(b_px.size(0)):
            x1, y1, x2, y2 = b_px[idx].tolist()

            if x2 <= x1 or y2 <= y1: continue

            color = plt.cm.jet(idx / max(1, b_px.size(0)))
            rect = Rectangle((x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

            ax.text(
                x1, max(0, y1-2), str(idx+1),
                fontsize=9, color='black', weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

            if idx < b_px.size(0) - 1:
                nx1, ny1, nx2, ny2 = b_px[idx+1].tolist()
                curr_center = ((x1+x2)/2, (y1+y2)/2)
                next_center = ((nx1+nx2)/2, (ny1+ny2)/2)
                if abs(curr_center[0]-next_center[0]) < W/2:
                    ax.annotate("", xy=next_center, xytext=curr_center, arrowprops=dict(arrowstyle="->", color=color, alpha=0.3))

    plt.axis("off")
    plt.show()

def check_sample_from_df(row, encoder_model):

    image_path = row['image_path']
    ocr_path = row['ocr_path']
    dataset_name = row.get('dataset', 'Unknown')

    print("\n" + "="*60)
    print(f"🖼️ ẢNH: {os.path.basename(image_path)}")
    print(f"📂 NGUỒN: {dataset_name}")
    print(f"📄 OCR PATH: {ocr_path}")

    out = encoder_model.load_ocr_features(image_path, ocr_path)

    if out is None:
        print("=> ❌ load_ocr_features trả về None.")
        return

    texts_after = out["texts"]
    boxes_after = out["boxes"] # Đã được sort và normalize

    joined_after = " ".join([str(t).strip() for t in texts_after if str(t).strip()])
    print("-" * 60)
    print(f"📝 TEXT READING ORDER ({len(texts_after)} boxes):")
    print(f"\"{joined_after[:500]}{'...' if len(joined_after) > 500 else ''}\"")
    print("-" * 60)

    _draw_boxes_in_order(image_path, boxes_after, title=f"Src: {dataset_name} | {os.path.basename(image_path)}")

# if 'final_train_df' in globals() and 'vision_ocr' in globals():
#     samples = final_train_df.sample(1)

#     for idx, row in samples.iterrows():
#         check_sample_from_df(row, vision_ocr)
# else:
#     print("⚠️ Cảnh báo: Không tìm thấy biến 'final_train_df' hoặc 'vision_ocr'. Hãy chạy các cell setup trước.")