# Bản đồ mã nghiên cứu ViSceT5

Luồng hiện hành huấn luyện trực tiếp VQA bằng cross-entropy trên câu trả lời,
cộng loss phụ TWC (tùy chọn `use_twc`). Không có pretrain; ITM và MLM đã bỏ.
Kênh sửa OCR (`ablation_use_ocr_aug`) nối bản OCR đã sửa (related) sau bản gốc.
Các trọng số backbone ViT5 và CLIP dùng để khởi tạo mô hình. Nhánh Visual Search
và ConvNeXtV2 đã được loại bỏ khỏi kiến trúc.
`models/modules/visual_search.py` vẫn được giữ làm source tham khảo, không import
trong luồng hiện hành. Feature ảnh/OCR vào concat trực tiếp, không qua projector.

## Luồng dữ liệu

Kịch bản YAML/notebook hiện tại: `ablation_use_ocr_aug=false`, `use_twc=false`,
`ablation_use_consformer=true`, `ablation_use_ocr=true` và có image.
OCR Origin qua ConsFormer với GroupAttention đã sửa, rồi fusion character/bbox/det/rec
và concat với question/image vào ViT5 encoder. Không có Related hay loss TWC.
Cờ ConsFormer được lưu trong checkpoint; checkpoint cũ mặc định vẫn bật nó.
Output YAML: `output/finetune_no_aug_no_twc_consformer_fixed`; notebook có hậu tố `groupfix`.

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

Mode hiện hành `ocr_aug_mode=separate`: các tensor `ocr_rel_input_ids`,
`ocr_rel_attention_mask`, `ocr_rel_to_word_map`, `ocr_rel_char_ids`, `ocr_rel_char_mask`
giữ bản Related riêng, với độ dài subword độc lập. Box/det/rec chỉ lưu một lần;
model nhân bản theo entry lúc fusion, offset mapping Related thêm W, giữ padding=-1.

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
hợp nhất các feature đó. Mọi nhánh đưa vào ViT5 phải có cùng chiều d_model.
Origin và Related đi qua hai lượt ConsFormer độc lập (chung trọng số), rồi concat
toàn bộ text feature trước khi cộng character/bbox/det/rec tương ứng. Không chọn
token lattice, không type embedding; từ giữ nguyên vẫn xuất hiện ở cả hai bản.

Encoder ViT5 nhận chuỗi ghép: question embeddings, CLIP patch tokens (nếu dùng ảnh),
OCR features (bản gốc rồi bản related, mức subword). Attention mask được ghép
cùng thứ tự. Decoder ViT5 sinh câu trả lời theo cách autoregressive.

## Kênh sửa OCR và TWC

`data/ocr_correction.py` (`OCRCorrector`): từ điển = term list + từ điển tiếng Việt
+ từ của câu hỏi/đáp án TẬP TRAIN. Hai cách sinh bản related (cùng độ dài, cùng thứ tự,
dùng chung box/det/rec với entry gốc):

- `correct` (xác định): giữ chuỗi đặc biệt/số/từ có
  trong từ điển; từ còn lại thay bằng ứng viên tốt nhất (dời dấu → đổi dấu → edit
  distance ≤ 2 cùng dạng không dấu hoặc phụ âm đầu). Hòa nhau phá bằng chính từ đó.
- `augment` (mode hiện hành, cả train và test): với xác suất 0.35, từ trong từ điển bị
  sửa một ký tự (nhãn 0.9), từ ngoài từ điển được sửa (0.9); còn lại giữ (1.0); cùng chữ
  kế thừa nhãn, khác chữ = 0, PAD = -1. Ngẫu nhiên cố định theo mẫu (tái lập được).

TWC (`OpenViVQAModel._twc`): cắt đầy đủ Origin và Related từ cùng output ViT5 encoder
mà decoder VQA sử dụng, rồi pool về entry theo mapping hai bản và chuẩn hóa L2.
Logits = cos/τ trên mọi cặp trong batch; exp(logit_scale)=1/τ với τ=0.07 cố định, không gradient/optimizer;
nhãn block-diagonal theo mẫu, cặp khác mẫu cùng (ảnh, chữ) kế thừa nhãn, hàng/cột PAD bỏ;
loss = sigmoid-BCE hai chiều trên ô hợp lệ (như code TWA). Tổng loss = VQA + λ·TWC.
YAML bật `use_twc=true`, `twc_feature=fused`, `twc_view=shared`. TWC truyền gradient
qua ViT5 encoder về cả hai lượt ConsFormer; decoder nhận gradient từ loss VQA.
Collator sinh cùng bản Related lúc train và eval/predict; chỉ train tạo nhãn TWC.
`concat` và `lattice` cũ được giữ cho checkpoint lịch sử, không dùng trong YAML/notebook.

## Module và ablation

| File | Vai trò |
| --- | --- |
| `models/modules/qa_clip.py` | CLIP với instruction ở nửa sau encoder; giữ cơ chế freeze CLIP gốc |
| `models/modules/ocr_consformer.py` | OCR encoder với GroupAttention |
| `models/modules/ocr_spatial.py` | Semantic OCR embedding và spatial utilities |
| `models/modules/ocr_encoder_feature.py` | Đọc OCR features và chuẩn hóa box |

Module tắt thì không được dựng (không tham số); mỗi nhóm module khởi tạo từ seed riêng
nên các module chung có cùng trọng số ban đầu giữa các cấu hình.
`ablation_use_image=False`: không có CLIP, encoder chỉ nhận [câu hỏi, OCR].
`ablation_use_qaclip=False` (mặc định): CLIP thường, không adapter QA-ViT.
`ablation_use_ocr=False`: baseline OCR text-only mức subword (embedding ViT5 → RMSNorm →
Linear khởi tạo đơn vị), không ký tự/box/det/rec.

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
không resume optimizer của kiến trúc cũ vì danh sách parameter đã thay đổi.
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
