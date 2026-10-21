# Đối chiếu toàn model khi VRAM giảm từ 32 GB xuống 12 GB

> Báo cáo lịch sử cho kiến trúc có Visual Search. Kiến trúc hiện tại đã gỡ toàn bộ
> ConvNeXt/ROI/crop tokens, giảm 27,866,496 parameter. Với backbone mặc định:
> Baseline không joint attention: total = 345,432,198; frozen = 85,799,424;
> learnable = 259,632,774. Nhánh OCR-image joint attention thêm ròng 7,088,640
> parameter (ba attention/pre-norm, thay ba norm OCR bằng hai norm fusion):
> total = 352,520,838; frozen = 85,799,424; learnable = 266,721,414.
> Geometry bias mới thêm 1,860 parameter trainable với 12 heads:
> total = 352,522,698; frozen = 85,799,424; learnable = 266,723,274.
> Khôi phục augmented OCR không thêm parameter vì hai lượt dùng chung OCR encoder.
> Với augmentation bật trong YAML, chuỗi encoder dài `229 + L_origin + L_aug`;
> khi tắt là `229 + L_ocr`. Chi phí activation/VRAM có thể tăng theo chuỗi mới.
> Công cụ profile đã được cập nhật cho
> kiến trúc mới; để tái hiện phép đo lịch sử dưới đây, dùng script ở commit `722f9fe`.
> Joint attention bổ sung dùng chuỗi `196 + W_ocr` (có padding), trước khi
> mapping OCR từ entry-level sang subword-level. Chưa đo VRAM trên GPU cho nhánh này.
> Geometry dùng additive attention mask riêng theo batch/head, có thể thay đổi kernel
> và activation memory; số parameter bổ sung không đại diện cho chi phí VRAM thực.

Baseline được kiểm tra là `d642998`, ngay trước commit refactor QAClip `b5f62ea`.
Bản đối chiếu là `aaa8a75`, đã chuyển attention sang `[visual, question]`.
Nếu lần chạy 32 GB dùng commit khác, cần đối chiếu đúng commit đó; kết quả dưới
đây không đại diện cho mọi phiên bản trong lịch sử repo.

## Tham số của toàn bộ kiến trúc

Đã dựng cả hai model trên meta device, sử dụng config thật của
`VietAI/vit5-base`, `openai/clip-vit-base-patch16` và
`facebook/convnextv2-tiny-22k-224`. Meta device dùng để kiểm tra kiến trúc mà
không cấp phát trọng số lớn; đây không phải phép đo VRAM hay kiểm tra giá trị
trọng số pretrained. Các config backbone được đọc từ Hugging Face ngày 2026-10-11.

| Module | Tổng tham số, cả hai bản | `requires_grad=True`, cả hai bản |
| --- | ---: | ---: |
| ViT5 | 225,950,976 | 225,950,976 |
| QAClip | 96,430,086 | 10,630,662 |
| Visual Search / ConvNeXtV2 | 27,866,496 | 27,866,496 |
| OCR Consformer + semantic + character | 20,093,568 | 20,093,568 |
| OCR-lite | 2,957,568 | 2,957,568 |
| **Toàn model** | **373,298,694** | **287,499,270** |

Tên, shape, dtype và `requires_grad` của từng parameter đều khớp. Danh sách
state-dict keys khớp; embedding ViT5 vẫn được dùng chung. Không có thay đổi
freeze ViT5, CNN hoặc OCR; CLIP backbone vẫn freeze và sáu block cuối có adapter
trainable. Gradient checkpointing mặc định vẫn tắt. YAML vẫn batch 4,
accumulation 2, AdamW, `fp16=False`, `bf16=False` theo mặc định.

Với OCR đầy đủ, OCR-lite không chạy nên 2,957,568 parameter không có gradient.
LayerNorm cuối ConvNeXt có thêm 1,536 parameter không có gradient, vì model lấy
`last_hidden_state`, không dùng pooled output đã qua LayerNorm. Cả hai điều này
đã có ở baseline. Chúng vẫn nằm trong nhóm optimizer do `requires_grad=True`,
nhưng Adam không tạo moments cho parameter có `grad=None`.

Theo đường forward này, số parameter dự kiến có gradient là 284,540,166. FP32
weights + gradients + hai Adam moments tương ứng khoảng 4.57 GiB, chưa tính
activations, temporaries, allocator, CUDA context hoặc workspace của kernels.
Đây là phép tính dung lượng tensor, không phải peak VRAM đo được.

## Kiểm tra thực thi của toàn pipeline

Đã nạp class QAClip và OpenViVQAModel trực tiếp từ source baseline, dựng model
tổng của mỗi bản với backbone nhỏ. QAClip dùng loader Hugging Face thật với cùng
checkpoint CLIP nhỏ; T5/CNN dùng constructor thật với config nhỏ. Sau đó copy
cùng state dict để tách khác biệt công thức khỏi khác biệt random initialization.

Batch gồm ảnh, câu hỏi, OCR token/word mapping, character, box, det/rec và answer.
Tắt dropout để đối chiếu FP32, bật cả ba nhánh, thử gate 0 và 0.3:

- Loss khớp ở cả hai thử nghiệm.
- Logits sai lệch tuyệt đối tối đa dưới `1e-6`.
- Gradient từng parameter sai lệch tuyệt đối tối đa `1.79e-7`.
- Các nhánh ViT5, QAClip, CNN, Consformer, semantic và character nhận gradient.
- Hai bản có cùng 172 tensor gradient và 172 Adam states trong cấu hình nhỏ.
- Optimizer chứa chính xác toàn bộ parameter có `requires_grad=True`.
- Shape, dtype và số lượt gọi từng nhánh đều khớp; ViT5 encoder vẫn chạy hai
  lần: contextual question rồi chuỗi đa phương thức. OCR vẫn ở mức subword.
- Saved-tensor hooks đếm storage ngoài parameter: 422,420 bytes trước và
  418,080 bytes sau ở gate 0. Đây là lượng storage được lưu cho backward của
  thử nghiệm CPU nhỏ, không phải peak allocation hoặc phép đo trên A100.

Không thấy graph bị cắt hoặc module trainable bị mất trong phạm vi kiểm tra.
Máy kiểm tra chỉ có PyTorch CPU, nên chưa tái hiện được hai số 32 GB và 12 GB.

## Những khác biệt code thực tế

Trong khoảng baseline → bản hiện tại, các file OCR, Visual Search, collator,
trainer, cấu hình và notebook không đổi. Model tổng chỉ thay lời gọi khởi tạo
projection bằng xử lý trong loader, và đổi offset lấy attention cho Visual
Search tương ứng thứ tự keys mới.

QAClip giữ kiến trúc và tham số, đồng thời sửa loader/checkpointing/mask, tính
queries chỉ cho visual tokens và đổi thứ tự joint keys. Với 197 visual tokens và
32 question tokens, riêng tensor QK của các block QA giảm từ `229 × 229` xuống
`197 × 229`, khoảng 14%. Không có căn cứ từ những kiểm tra này để quy mức giảm
20 GB của toàn pipeline cho thay đổi đó.

Điểm cần đo trên batch thật: `ocr_max_scene_text=180` giới hạn số mục OCR,
không giới hạn tổng số subword. Collator cho phép mỗi mục tối đa 25 subword;
tổng lý thuyết có thể tới 4,500. Với các backbone mặc định, độ dài chuỗi ViT5
đa phương thức là `32 + 196 + L_ocr + 49 + 1 = 278 + L_ocr`.
Consformer và ViT5 self-attention chứa tensor tăng theo bình phương độ dài
chuỗi. Cùng batch size và config chưa đảm bảo cùng lượng activations nếu các
batch có độ dài OCR khác nhau. Collator này cũng không đổi qua refactor QAClip.

## Đo trực tiếp trong process training trên Colab

`scripts/profile_training_memory.py` chạy Trainer thật trên một batch cố định,
mặc định ba optimizer steps với accumulation/precision/optimizer từ YAML.
Nó không sửa model hay thay đổi lệnh full training. Lệnh chẩn đoán tắt eval/save,
dùng worker 0 và lặp lại cùng batch để tránh nhiễu từ dữ liệu/prefetch.

Chạy bản hiện tại trong một process mới:

```bash
python scripts/profile_training_memory.py configs/finetune.yaml \
  --audit_batch_file /content/vqa-audit-batch.pt \
  --output_dir /content/vqa-memory-current
```

Lần đầu tạo batch từ dữ liệu; các lần sau đọc đúng file đó, gồm toàn bộ
token/mapping/features/ảnh. Có thể dùng `--model_name_or_path` để nạp trọng số
model. Lệnh này đo optimizer khởi tạo mới, không resume trạng thái Trainer.
Chỉ đọc batch `.pt` do chính phép đo này tạo.

Để chạy baseline với cùng batch và cùng môi trường, từ checkout hiện tại:

```bash
git worktree add --detach /content/ViSceT5-before d642998
cp scripts/profile_training_memory.py /content/ViSceT5-before/scripts/
cd /content/ViSceT5-before
python scripts/profile_training_memory.py configs/finetune.yaml \
  --audit_batch_file /content/vqa-audit-batch.pt \
  --output_dir /content/vqa-memory-before
```

Các báo cáo:

- `manifest.json`: commit, source hashes, phiên bản thư viện, GPU, config thực
  tế, batch hash/shape/dtype, checkpointing và từng parameter.
- `memory_trace.json`: allocated/reserved và peak theo optimizer step; các
  mốc trước/sau forward, sau backward, trước/sau optimizer; shape/dtype/output
  gradient của từng nhánh. Peak được reset đầu mỗi optimizer step, không
  `empty_cache()` và không reset giữa các microsteps.
- `parameters_after_optimizer.json`: từng parameter có gradient hay không,
  có trong optimizer hay không, bytes parameter/gradient/Adam state, tổng theo
  module. Báo cáo cuối được ghi sau optimizer, trước khi Trainer xóa gradient.

So sánh bước 2–3, sau khi Adam states đã tạo ở bước 1. Các chỉ số là bytes;
chia `2**30` để đổi sang GiB. Các mốc có synchronize nên phép đo này không
dùng để benchmark tốc độ. GPU memory được đo trong chính process Trainer,
kể cả khi gọi lệnh từ notebook chạy subprocess.

`memory_allocated` phản ánh tensor còn sống; `memory_reserved` gồm vùng do
allocator giữ lại. Chỉ số GPU tổng còn có thể gồm cache/context/process khác.
Xem [PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management).
Chưa có hai trace A100 để xác định nguyên nhân của chênh lệch quan sát được.
