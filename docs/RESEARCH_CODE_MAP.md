# Bản đồ mã nghiên cứu ViSceT5

Luồng hiện hành huấn luyện trực tiếp VQA bằng cross-entropy trên câu trả lời.
Các mục tiêu ITM, MLM, TWC và việc sinh OCR related/augmented đã được loại bỏ.
Các trọng số backbone ViT5 và CLIP dùng để khởi tạo mô hình. Nhánh Visual Search
và ConvNeXtV2 đã được loại bỏ khỏi kiến trúc.
`models/modules/visual_search.py` được giữ làm source tham khảo, không export/import
trong luồng hiện hành; các cờ bật/tắt nhánh cũng đã bỏ.

## Luồng dữ liệu

`configs/data/*.yaml` → `scripts/prepare_dataset.py` → `DatasetHubLoader`
→ CSV train/validation/test → `ViT5VQADataset` → `ViT5VQADataCollator`.

Mỗi sample gồm đường dẫn ảnh, đường dẫn OCR tùy chọn, câu hỏi và câu trả lời.
OCR loader đọc `.npy`, lọc confidence, chuẩn hóa box về [0, 1] và sắp xếp
theo thứ tự đọc. Collator chuẩn hóa văn bản, loại noise/duplicate và giữ nguyên
sự tương ứng giữa text, box, detection feature và recognition feature.
Không có OCR hợp lệ thì dùng một entry `<unk>` với feature bằng 0.

Các tensor OCR chính:

| Tensor | Shape | Ý nghĩa |
| --- | --- | --- |
| `ocr_input_ids` | B × L | Subword của OCR gốc |
| `ocr_attention_mask` | B × L | Subword hợp lệ |
| `ocr_to_word_map` | B × L | Index entry OCR; padding = -1 |
| `ocr_char_ids` | B × W × C | ID ký tự |
| `ocr_char_mask` | B × W × C | Ký tự hợp lệ |
| `ocr_word_mask` | B × W | Entry OCR hợp lệ |

L là số subword được pad trong batch; W là số entry được pad, không vượt
`ocr_max_scene_text`; C là `char_max_num`. Box/det/rec trong `ocr_info` được
pad đến cùng W. `labels` là token câu trả lời; padding = -100. Batch inference
không có câu trả lời không tạo `labels`.

## Luồng mô hình

`models/openvivqa_model.py` chứa `OpenViVQAModel` và ba bước công khai:

1. `encode()` tạo encoder memory đa phương thức.
2. `forward()` dùng memory để tính decoder loss/logits khi có labels.
3. `generate()` sinh câu trả lời; có thể nhận memory đã encode.

Câu hỏi được encode bởi ViT5 để làm instruction cho QACLIP. CLIP trả về patch
tokens; model tổng không yêu cầu xuất attention maps và không trích xuất ROI/CNN.

Chi tiết đối chiếu QAClip với repo QA-ViT gốc, khởi tạo adapter và các điều chỉnh
được ghi tại [QA_VIT_REFACTOR.md](QA_VIT_REFACTOR.md).

OCR Consformer encode chuỗi subword qua GroupAttention. Mapping subword→entry
gắn box, det/rec và character features với từng subword. SemanticOCREmbedding
hợp nhất các feature đó. Mapping token-level OCR không thay đổi.

Ngay trước concat, `image_projector` là Linear từ chiều hidden CLIP sang ViT5
d_model; `ocr_projector` là Linear độc lập từ d_model sang d_model, sau Semantic
OCR hoặc OCR-lite. Cả hai khởi tạo random mặc định PyTorch, được train và lưu trong
checkpoint. Đây là cầu nối học được giữa các không gian feature, kể cả khi cùng
chiều 768. Chúng không pool/đổi thứ tự token; OCR padding được mask lại sau projection.
Projector một lớp là lựa chọn nhẹ hơn MLP hai lớp ReLU của QA-ViT; nhánh OCR là
phần mở rộng của ViSceT5. Không dùng trọng số pretrained ViT5 cho projector.

Encoder ViT5 nhận chuỗi ghép: question embeddings, projected CLIP patch tokens,
projected OCR features, một token trung bình của các projected patches. Attention mask được ghép
cùng thứ tự. Decoder ViT5 sinh câu trả lời theo cách autoregressive.

## Module và ablation

| File | Vai trò |
| --- | --- |
| `models/modules/qa_clip.py` | CLIP với instruction ở nửa sau encoder; giữ cơ chế freeze CLIP gốc |
| `models/modules/ocr_consformer.py` | OCR encoder với GroupAttention |
| `models/modules/ocr_spatial.py` | Semantic OCR embedding và spatial utilities |
| `models/modules/ocr_encoder_feature.py` | Đọc OCR features và chuẩn hóa box |
| `models/modules/projectors.py` | Hai Linear projector riêng cho ảnh/OCR trước encoder ViT5 |

Tắt QACLIP nghĩa là bỏ instruction câu hỏi, vẫn chạy CLIP. Tắt OCR module nghĩa là dùng encoder lite
cho text/character/box, vẫn nhận thông tin OCR. Đây là các baseline ablation
của repository, cần giữ ý nghĩa này khi thiết kế thí nghiệm tiếp theo.

## Training, evaluation và checkpoint

`training/common.py` dùng chung việc đọc YAML/CLI, chuẩn bị dataset, nạp checkpoint
và tạo collator. `training/finetune.py` cấu hình Trainer và lưu kết quả.
`training/trainer.py` chứa `VQATrainer`: loss VQA, generation từ encoder memory
đã tính, kiểm tra gradient hữu hạn và ghi metrics JSONL.
`training/metrics.py` chỉ còn EM, token F1, BLEU và CIDEr.
`training/evaluate.py` nạp checkpoint và đánh giá validation/test.

Checkpoint dùng định dạng Hugging Face thông thường, gồm config, model weights
(có hỗ trợ sharding), tokenizer và image processor. Warm-start nạp model weights;
resume từ `checkpoint-*` còn khôi phục optimizer, scheduler và Trainer state.
Checkpoint có Visual Search có thể dùng để warm-start những trọng số còn lại;
hai projector mới sẽ được khởi tạo nếu checkpoint chưa có. Không resume optimizer
của kiến trúc cũ vì danh sách parameter đã thay đổi.
Các checkpoint cũ chứa head hoặc metadata của những mục tiêu đã bỏ không phải
checkpoint chuẩn của luồng mới; các tensor batch cũ cũng đã đổi sang API OCR ở trên.

`run_pipeline.py`, các shell launcher và notebook Colab chỉ gọi luồng VQA.
Các tiện ích chuẩn bị dữ liệu/vocabulary được giữ để phục vụ nghiên cứu tiếp.

## Phạm vi kiểm chứng

`tests/test_vqa_pipeline.py` kiểm tra bằng ảnh/OCR tổng hợp và backbone ngẫu nhiên
kích thước nhỏ: alignment, padding, OCR rỗng, dữ liệu không nhãn, 4 tổ hợp ablation,
loss/backward, cached generation, checkpoint sharding và train/evaluate/resume.
Việc này xác minh luồng mã; chất lượng nghiên cứu và tài nguyên GPU cần được đo
qua thí nghiệm trên dataset và backbone thực tế.
