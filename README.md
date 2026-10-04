# OpenViVQA

**Vietnamese Visual Question Answering** — A multimodal model combining ViT5, CLIP, OCR Consformer, and Visual Search.

## Architecture

```
OpenViVQAModel
├── ViT5 (VietAI/vit5-base)            — Encoder-Decoder language backbone
├── QACLIPEncoder                       — Fast visual stream: CLIP ViT-B/16 @ 336 with MMCLIPAttention
│   └── InstructCLIPEncoder             — Late-fusion: query-guided cross-attention layers
├── MRA (Multi-Resolution Adapter)      — Slow visual stream: ConvNeXtV2-Base @ 1024 (Feast Your Eyes)
│   └── Cross-Resolution Adapters       — Injects high-res fine-grained spatial features into ViT
└── OCR Consformer                      — OCR token sequence & 2D spatial position embedding
```

**Training Objectives (Pretrain)**
| Objective | Description | Target Component |
|---|---|---|
| **PreSTU SplitOCR** *(Primary)* | Generative text-infilling from image (Prefix → Target) | ViT5 Decoder (Direct Seq2Seq generation) |
| **TWC** *(Auxiliary)* | Token-Word Contrastive loss (aligned from TWA paper) | Cross-modal alignment layer |
| **ITM / ITC** *(Auxiliary)* | Image-Text Matching & InfoNCE Contrastive (pollute detection) | Vision-Language representation space |
| **MLM** *(Legacy)* | Masked Language Modelling on question + OCR tokens | Encoder text head |

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

#### Bước A: Pretrain (Đọc Hiểu Chữ Cảnh Thực & Căn Chỉnh Thị Giác - Ngôn Ngữ)

Giai đoạn **Pretrain** là bước huấn luyện nền móng quan trọng nhằm trang bị cho mô hình hai năng lực cốt lõi:
1. **Scene-Text Visual Grounding**: Nhìn vào hình ảnh thực tế để phát hiện, định vị và đọc chính xác chữ cảnh thực (biển hiệu, thực đơn, nhãn sản phẩm).
2. **Vision-Language Alignment**: Căn chỉnh biểu diễn không gian đặc trưng giữa hình ảnh và ngôn ngữ tiếng Việt trước khi finetune.

##### 1. Thiết Kế Đột Phá: MRA (Multi-Resolution Adapter)

Thay vì cắt ảnh cục bộ (crop) bằng module Visual Search rời rạc cũ (dễ gây đứt gãy ngữ cảnh và tiêu tốn VRAM), mô hình áp dụng kiến trúc **MRA** (dựa trên nguyên lý *Feast Your Eyes*):
* **Fast Stream (Toàn cảnh ngữ nghĩa)**: QA-CLIP ViT-B/16 @ 336×336 với cơ chế truy vấn câu hỏi động (*MMCLIPAttention*).
* **Slow Stream (Chi tiết siêu phân giải)**: ConvNeXtV2-Base @ 1024×1024, trích xuất cấu trúc không gian chi tiết và các ký tự nhỏ li ti trong cảnh thực tế.
* **MRA Cross-Fusion Adapters**: Bơm trực tiếp các bản đồ đặc trưng độ phân giải cao từ ConvNeXtV2 vào 3 block Transformer cuối của ViT.
* **Differential Learning Rate (`VISION_LR_SCALE=0.2`)**: Mở băng toàn bộ ViT (`vision_unfreeze_last_n: -1`) nhưng giảm LR thị giác $\times 0.2$ để thích ứng với chữ tiếng Việt mà không phá hủy biểu diễn tiền huấn luyện của CLIP/ConvNeXt.

##### 2. Mục Tiêu Huấn Luyện Cốt Lõi: PreSTU SplitOCR

Khác với mục tiêu Masked Language Modeling (MLM) kiểu BERT truyền thống (chỉ rèn luyện đầu encoder và bị vứt bỏ khi finetune), phương pháp **PreSTU SplitOCR** rèn luyện trực tiếp kiến trúc sinh chuỗi của Decoder:
* **Cơ chế SplitOCR**: Chuỗi văn bản OCR trong ảnh được chia ngẫu nhiên thành hai phần rời rạc: `[Prefix, Target]`.
* **Nhiệm vụ của mô hình**: Nhìn vào hình ảnh kết hợp với phần chữ `Prefix` gợi ý để tự động sinh ra chính xác phần chữ `Target` còn lại.
* **Curriculum Training**: 
  - `--pretrain_split_mode sequential`: Tách chuỗi theo thứ tự đọc tự nhiên của con người.
  - `--pretrain_full_ocr_prob 0.2`: Trong 20% trường hợp, `Prefix` bị bỏ trống hoàn toàn, buộc mô hình phải đọc và sinh 100% chuỗi chữ chỉ từ hình ảnh đơn thuần.

##### 3. Bảng Chế Độ Hàm Mất Mát (`loss_ablation_mode`)

| Chế độ | Hàm mất mát kích hoạt | Ý nghĩa & Khuyến nghị |
|---|---|---|
| **`prestu`** *(Khuyến nghị)* | **PreSTU SplitOCR** | Thuần sinh văn bản qua Decoder ViT5 từ ảnh. Khớp hoàn hảo với tác vụ sinh câu trả lời VQA downstream. |
| **`gen_all`** | **PreSTU + TWC + ITM (+ ITC)** | Kết hợp vừa học sinh chữ ở Decoder vừa duy trì các đầu căn chỉnh đối sánh (TWC/ITM/ITC) ở Encoder. |
| **`all`** *(Cũ)* | **MLM + TWC + ITM** | Huấn luyện che từ truyền thống trên Encoder (không rèn luyện trực tiếp Decoder). |

##### 4. Dữ Liệu Huấn Luyện Pretrain
* `--dataset_name "VinText,EVJVQA"`:
  - **VinText**: Tập dữ liệu chữ cảnh thực lớn nhất cho tiếng Việt (biển hiệu, thực đơn, bao bì đường phố).
  - **EVJVQA**: Tập dữ liệu hỏi đáp thị giác thực tế đa ngữ.

##### 5. Quy Trình Chạy Pretrain

###### Bước 1: Smoke / Mock Test (Kiểm tra pipeline an toàn)
Chạy thử vài step với dữ liệu nhỏ để kiểm tra cấu trúc pipeline, không lưu checkpoint đè lên đĩa và không push lên Hub:
```bash
python training/pretrain.py configs/pretrain.yaml \
    --dataset_name "VinText,EVJVQA" \
    --ablation_use_mra True \
    --ablation_use_vs False \
    --mra_high_res 1024 \
    --pretrain_gen_only True \
    --pretrain_split_mode sequential \
    --pretrain_full_ocr_prob 0.2 \
    --vision_unfreeze_last_n -1 \
    --smoke_test True
```

###### Bước 2: Huấn Luyện Full Pretrain (6 Epochs)
Huấn luyện đầy đủ với bộ siêu tham số tối ưu (Effective Batch Size = 16):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VISION_LR_SCALE=0.2 python training/pretrain.py configs/pretrain.yaml \
    --dataset_name "VinText,EVJVQA" \
    --clip_vision_name openai/clip-vit-base-patch16 \
    --clip_image_size 336 \
    --vs_backbone facebook/convnextv2-base-22k-384 \
    --ablation_use_mra True \
    --ablation_use_vs False \
    --mra_high_res 1024 \
    --pretrain_gen_only True \
    --pretrain_split_mode sequential \
    --pretrain_full_ocr_prob 0.2 \
    --vision_unfreeze_last_n -1 \
    --num_train_epochs 6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --gradient_checkpointing True \
    --bf16 True \
    --tf32 True \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 2 \
    --output_dir ./output/pretrain_mra \
    --logging_dir ./output/pretrain_mra/logs
```

##### 6. Cơ Chế Tự Động Resume & Bảo Vệ Tiến Độ (2-Tier Resume)

* ☁️ **Auto-push lên Hugging Face Hub**: Mỗi khi kết thúc một epoch, checkpoint sẽ tự động được đồng bộ lên repo Hugging Face (`HF_REPO=Kus669/ViSceT5-mra-pretrain-ver4`).
* ♻️ **Tự động khôi phục (Resume 2 Lớp)**:
  - Khi chạy lại lệnh (hoặc máy ảo bị ngắt kết nối), hệ thống tự động so sánh số step giữa đĩa cục bộ (`./output/pretrain_mra`) và repo từ xa trên Hugging Face.
  - Tự động tải bản mới nhất về nếu remote có step cao hơn (hoặc trên máy ảo mới mở).
  - Tự động tiếp tục từ đĩa cục bộ nếu đã có sẵn checkpoint hợp lệ mà không tốn thời gian tải lại.
  - Tự động phát hiện và bỏ qua các checkpoint mock từ smoke test.

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
