# OpenViVQA

**Vietnamese Visual Question Answering** — A multimodal model combining ViT5, CLIP, OCR Consformer, and Visual Search.

## Architecture

```
OpenViVQAModel
├── ViT5 (VietAI/vit5-base)            — Encoder-Decoder language backbone
├── QACLIPEncoder                       — CLIP with query-guided (MMCLIPAttention) vision encoder
│   └── InstructCLIPEncoder             — Late-fusion: plain CLIPEncoderLayer for first half,
│       ├── CLIPEncoderLayer            — MMCLIPEncoderLayer for second half
│       └── MMCLIPEncoderLayer
├── VisualSearch                        — ConvNeXtV2-based attention crop
└── OCR Consformer
    ├── OCREncoder (GroupAttention)     — Encodes OCR token sequences with neighbor attention
    ├── SpatialCirclePosition           — 2D distance-aware spatial position embedding
    └── SemanticOCREmbedding            — Fuses bounding box + text representations
```

**Training Objectives (Pretrain)**
| Objective | Description |
|-----------|-------------|
| **MLM** | Masked Language Modelling on question + OCR tokens |
| **TWC** | Token-Word Contrastive loss (aligned from TWA paper) |
| **ITM** | Image-Text Matching / Pollute detection head |

## Repository Structure

```
openvivqa/
├── configs/
│   ├── base_config.py          # SEED, OUTPUT_PATH, configure_env()
│   ├── model_config.py         # OpenViVQAConfig (PretrainedConfig)
│   └── ocr_config.py           # Default OCR encoder config
│
├── data/
│   ├── dataset_hub.py          # DatasetHubLoader — download & prepare datasets
│   ├── dataset.py              # ViT5VQADataset (PyTorch Dataset)
│   ├── collator.py             # ViT5VQADataCollator (TWC+MLM+ITM)
│   ├── vocab.py                # Char vocab, text normalisation, OCR augmentation utils
│   ├── ocr_utils.py            # Vision_Encode_Ocr_Feature, reading-order sort
│   ├── data_loader.py          # load_dataset_final()
│   └── eda.py                  # EDA analyzers
│
├── models/
│   ├── openvivqa_model.py      # OpenViVQAModel (main model)
│   └── modules/
│       ├── attention.py        # LayerNorm, FC, MLP, AoA, SAoA, GAoA, SGAoA
│       ├── ocr_consformer.py   # OCREncoder, GroupAttention
│       ├── ocr_encoder_feature.py  # Vision_Encode_Ocr_Feature
│       ├── ocr_spatial.py      # SpatialCirclePosition, SemanticOCREmbedding
│       ├── qa_clip.py          # QACLIPEncoder, MMCLIPAttention
│       └── visual_search.py    # VisualSearch (ConvNeXtV2)
│
├── training/
│   ├── finetune.py             # Full finetune training loop (Seq2SeqTrainer)
│   ├── metrics.py              # compute_metrics(), BLEU/CIDEr evaluation
│   └── evaluate.py             # Evaluation / prediction pipeline
│
├── utils/
│   ├── misc.py                 # SET_SEED(), pick_consistent_indices()
│   ├── model_utils.py          # print_trainable_params(), safe_download_weights()
│   ├── visualization.py        # OCR box visualization, sample display
│   └── debug_tools.py          # Collator inspection, dummy batch generation
│
└── scripts/
    ├── prepare_dataset.py      # Download & prepare datasets
    └── init_model.py           # Download backbone weights, initialize model
```

## Hướng Dẫn Chạy (Quick Start)

Dự án này được thiết kế để chạy trơn tru trên mọi Server thông qua **HuggingFace ArgumentParser**. Các tham số huấn luyện được gom gọn trong các file YAML tại thư mục `configs/`.

### 1. Chuẩn bị Môi trường
Cài đặt các thư viện cần thiết:
```bash
bash setup.sh
```

### 2. Khởi tạo Dataset
Trước khi huấn luyện, bạn cần kéo dữ liệu về Server. Bằng cách gọi lệnh dưới, hệ thống sẽ tự động gdown từ Drive nếu thư mục dữ liệu chưa có:
```bash
python scripts/prepare_dataset.py --data_dir ./datasets
```
*Lưu ý: Mặc định script sẽ dùng `configs/data/ViTextVQA.yaml`.*

### 3. Khởi tạo Model (Tùy chọn)
Script này nhằm tải các weights gốc (`VietAI/vit5-base` & `openai/clip`) về bộ nhớ đệm HuggingFace cục bộ và chạy thử một lượt forward pass để đảm bảo cấu trúc model khởi tạo thành công không bị Out-Of-Memory.
```bash
python scripts/init_model.py
```

### 4. Huấn Luyện (Training Pipeline)

#### Bước A: Pretrain
Giai đoạn này giúp các module chuyên biệt (OCR Consformer, Visual Search) làm quen với hình ảnh và văn bản qua 3 mục tiêu Loss: MLM, ITM, và TWC.
Mọi cấu hình nằm trong file `configs/pretrain.yaml`.
```bash
python training/pretrain.py configs/pretrain.yaml
```
- **Tinh chỉnh Config:** Các chế độ `loss_ablation_mode` hỗ trợ gồm: `"all"` (đầy đủ), `"only_itm_mlm"` (tắt TWC & OCR Aug), và `"only_twc_ocr_aug"` (tắt MLM & ITM).

#### Bước B: Finetune (Nối tiếp Pretrain)
Khi Pretrain hoàn tất, bạn lấy trọng số đó để finetune trực tiếp cho tác vụ sinh câu trả lời VQA.
Trong file `configs/finetune.yaml`, hãy chắc chắn rằng tham số `model_name_or_path` trỏ đúng vào thư mục Pretrain:
- `model_name_or_path: "./output/pretrain_ckpt_base"` (hoặc một thư mục checkpoint cụ thể)
- `loss_ablation_mode: "all"` (nên để khớp với lúc pretrain để đồng bộ cờ bật/tắt OCR Augmentation).
```bash
python training/finetune.py configs/finetune.yaml
```

#### Bước C (Tuỳ Chọn): Finetune Không Qua Pretrain (From Scratch)
Nếu muốn huấn luyện Finetune ngay từ đầu (bỏ qua Pretrain), bạn chỉnh sửa `configs/finetune.yaml` như sau:
1. Đặt `model_name_or_path: ""` (Bỏ rỗng để hệ thống tải backbone mặc định thay vì lấy checkpoint).
2. Thiết lập bật/tắt các module bạn muốn ablation:
   - `ablation_use_qaclip: true`
   - `ablation_use_vs: true`
   - `ablation_use_ocr: true`
```bash
python training/finetune.py configs/finetune.yaml
```

### 5. Tự động hóa và Cấu hình Biến môi trường (Vast.ai)

Nếu bạn chạy trên các Server Cloud như Vast.ai, bạn có thể sử dụng file [run_all.sh](file:///c:/Users/Admin/Workspace/openvivqa/run_all.sh) để tự động hóa hoàn toàn quá trình tải thư viện, dựng môi trường ảo, và chạy huấn luyện.

#### Các biến môi trường cần cấu hình trong `run_all.sh`:
*   `HF_TOKEN`: Token tài khoản Hugging Face của bạn (cần quyền **Write**) để tự động tải checkpoint lên Hub.
*   `HF_REPO`: Đường dẫn repo của Hugging Face Hub (ví dụ: `Kus669/ViSceT5-pretrain`).
*   `STAGE`: Giai đoạn huấn luyện. Chọn `"pretrain"` hoặc `"finetune"`.
*   `MOCK_TEST`: Thiết lập `"true"` để chạy thử nhanh (Smoke Test với 8 dòng dữ liệu và 3 steps) nhằm kiểm tra lỗi đường ống dẫn, hoặc `"false"` để chạy thật.

#### Cách chạy:
1. Mở file [run_all.sh](file:///c:/Users/Admin/Workspace/openvivqa/run_all.sh) và cập nhật token thật vào biến `export HF_TOKEN="YOUR_HF_TOKEN"`.
2. Cấp quyền thực thi và khởi chạy file script dưới nền:
   ```bash
   bash run_all.sh
   ```
3. Script sẽ chạy ngầm và xuất toàn bộ nhật ký huấn luyện ra file `train_execution.log`. Để theo dõi tiến trình chạy trực tiếp, sử dụng lệnh:
   ```bash
   tail -f train_execution.log
   ```

### 6. Resume Huấn Luyện (Tiếp tục khi bị gián đoạn)
Trong cả `pretrain.yaml` và `finetune.yaml`, bạn có thể dùng một trong hai cách để tiếp tục train nếu server bị sập:
- `resume_from_checkpoint: "./output/pretrain/checkpoint-1000"` (Trỏ thẳng vào ổ đĩa cục bộ).
- `resume_checkpoint_id: "ID_TRÊN_DRIVE"` (Nếu checkpoint nằm trên Google Drive dạng zip, hệ thống sẽ tự tải, giải nén và resume chuẩn xác số epoch/step).

## Key Dependencies
| Package | Version |
|---------|---------|
| transformers | 4.45.2 |
| peft | 0.13.1 |
| accelerate | 0.34.2 |
| torch | ≥2.0.0 |

## Notes
- **OCR features** are pre-computed `.npy` files. See `data/ocr_utils.py` for the loading format.
- **Term vocabulary** for TWC augmentation should be placed at `term_vocab_path` in config.
- Ablation switches: set `ablation_use_qaclip`, `ablation_use_vs`, `ablation_use_ocr` in `OpenViVQAConfig` to toggle individual modules.

## Bug Fixes vs Notebook
- `_encode_ocr_features` was accidentally dedented to module scope in the original notebook — fixed as a proper `OpenViVQAModel` method in `models/openvivqa_model.py`.
