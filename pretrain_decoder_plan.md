# Kế hoạch: Pretrain DECODER bằng mục tiêu sinh (đòn bẩy #1)

> Bối cảnh đã xác minh (2026-07-03):
> - **Bước 0 PASSED** — finetune nạp `806/809` tensor, `missing=3` là bộ tied-embedding
>   (`vit5.encoder.embed_tokens`, `vit5.decoder.embed_tokens`, `vit5.lm_head`) bị safetensors
>   dedup; `vit5.shared.weight` đã nạp + `tie_weights()` (`openvivqa_model.py:220`) → embedding/lm_head
>   THỰC SỰ là trọng số pretrain. Warm-start KHÔNG phải no-op.
> - Nguyên nhân from-pretrain ≈ from-scratch là **phương pháp**, bám paper TWA (3503161.3547977):
>   TWA trả lời bằng **pointer network** trỏ vào `z_ocr` (§3.7) nên pretrain (làm `z_ocr` bền lỗi)
>   chuyển giao trực tiếp. ViSceT5 trả lời bằng `vit5.generate()` (`openvivqa_model.py:1117`) — decoder
>   sinh subword — mà decoder **gần như không được pretrain** (answer labels toàn -100).

## Mục tiêu
Dạy **decoder** sinh scene-text grounded ngay từ pretrain, để warm-start có ích cho `generate()` ở finetune.
Hai tác vụ sinh (dùng chung encoder đa nhánh hiện có, chỉ thêm nhãn cho decoder):

1. **Read-scene-text (grounded)**: encoder = prompt cố định ("đọc chữ trong ảnh") + ảnh + OCR-as-feature;
   decoder sinh chuỗi OCR token theo thứ tự vị trí. → buộc decoder đọc từ ảnh/OCR-feature.
2. **OCR-denoise generation**: encoder nhận OCR **đã nhiễu** (augmentation §3.4 sẵn có); decoder sinh OCR
   **sạch**. → bản seq2seq-hoá đúng tinh thần error-tolerance của TWA, đặt vào đúng decoder.

## Nguyên tắc chống rò rỉ (để #1 thực sự grounding)
- **Không nhồi OCR-as-text vào encoder** khi đích cần sinh là chính OCR đó → nếu không, decoder chỉ copy.
- **Đồng bộ input pretrain ≈ finetune**: encoder thấy question/prompt-only + ảnh + OCR-feature
  (không phải `question+OCR` dạng text như MLM hiện tại).
- TWC/ITM/MLM **hạ xuống regularizer phụ** cho encoder, không còn là trục chính.

## Điểm chạm code (dự kiến)

### 1. `data/collator.py` — nhánh `self.pretrain`
- Hiện: tạo `mlm_input_ids`, `cmb_text_mask_label`, và `labels = lab.clone().fill_(-100)` (decoder không luyện).
- Thêm: nhãn sinh cho decoder.
  - Read-scene-text: `gen_labels = tokenizer(" ".join(ocr_tokens_sorted_by_pos))`.
  - OCR-denoise: encoder input dùng nhánh OCR **nhiễu** (đã có `rel_ocr`/`twa` tensors); `gen_labels = clean OCR`.
  - `encoder_prompt_ids` = prompt cố định (hoặc question) — KHÔNG kèm OCR-as-text.
- Xuất thêm khoá batch: `gen_labels`, (tuỳ) `gen_task_id` để model biết đang read vs denoise.

### 2. `models/openvivqa_model.py` — nhánh `if self.pretrain:` (~897–1054)
- Giữ MLM/ITM/TWC như hiện tại (regularizer).
- Thêm 1 lượt decoder sinh: `outputs_gen = self.vit5(encoder_outputs=enc_out, attention_mask=fused_mask,
  labels=gen_labels, ...)` → `out_dict["gen_loss"] = outputs_gen.loss`.
- Lưu ý: encoder input cho lượt gen phải là biến thể prompt-only (xem mục 1) — có thể cần encode riêng
  hoặc tái dùng `enc_out` nếu ta chuyển toàn bộ pretrain sang input prompt-only (khuyến nghị: chuyển hẳn).

### 3. `training/pretraine_loss.py` — `ViT5PretrainLoss.forward`
- Đọc `model_output["gen_loss"]`; cộng vào `total_loss` theo mode.
- Thêm mode mới `gen` / `gen_all`: `total = gen_loss + w_mlm*mlm + w_itm*itm + w_twc*twc`
  (khởi đầu: `w_mlm=w_itm=w_twc=1.0`, `gen` trọng số 1.0). Log `loss_gen`.
- Thêm metric `gen`: token-accuracy hoặc để trainer tự tính eval loss.

### 4. Cấu hình & env
- `configs/pretrain.yaml`: thêm `loss_ablation_mode` chọn được `gen_all`.
- `run_pipeline.py`: passthrough env (đã có cơ chế) — thêm 1 knob nếu cần (ví dụ `PRETRAIN_OBJECTIVE`).
- Chạy Colab: `STAGE=pretrain PRETRAIN_OBJECTIVE=gen_all bash run_all_colab.sh` (đúng thói quen bash-only).

## Đối chứng (bắt buộc)
Cùng seed/steps, so 3 cấu hình finetune:
- from-scratch,
- from-pretrain **cũ** (MLM+ITM+TWC),
- from-pretrain **mới** (gen_all).
Kỳ vọng: (mới) > (cũ) ≈ (scratch). Nếu (mới) không hơn → xem lại chống-rò-rỉ / input alignment.

## Ghi chú tránh lặp lỗi cũ
- KHÔNG chặn attention gốc↔related (đã thử & revert — xem memory `twc-must-match-notebook-paper`).
- Giữ NaN grad guard trong `training_step` (không có hook `_clip_grad_norm`).
- `twc_leak_analysis.md` phần Fix #3 đã lỗi thời, bỏ qua.

## Dữ liệu thực nghiệm (ViTextVQA train, 35.159 dòng; mẫu 6.000 — phân tích 2026-07-03)
- Độ dài đáp án: **1 từ chỉ 15.9%**; 2 từ 22.8%, 3 từ 19.0%, 4 từ 11.5%, 5 từ 9.1%; đuôi dài tới >12 từ.
  → Đáp án **mang tính SINH thật sự** (đa số nhiều từ) ⇒ kiến trúc seq2seq là đúng; KHÔNG nên ép về pointer.
- Đáp án là **substring nguyên văn** của OCR concat: **chỉ 11.6%**.
- **Toàn bộ từ** trong đáp án có mặt trong tập token OCR: **39.1%**.
- Đáp án **không có từ nào** trong OCR (suy luận thuần: màu/đếm/có-không/nội suy): **19.6%**.

**Hệ quả thiết kế (quan trọng):**
- read-scene-text KHÔNG phải để "bắt chước đáp án" (chỉ ~12% verbatim) mà là **pretext GROUNDING**:
  buộc decoder học đọc & sinh text tiếng Việt grounded từ ảnh+OCR — đúng năng lực đang thiếu.
  Vì ~40% đáp án cấu thành từ token OCR, năng lực đọc grounded chuyển giao đáng kể.
- OCR-denoise trực tiếp đánh vào giả thuyết trung tâm của TWA (lỗi OCR) và hỗ trợ chính ~40% đó.
- Target read-scene-text = OCR tokens nối theo **thứ tự đọc** (sort theo `boxes`: trên→dưới, trái→phải),
  không phải theo đáp án.
- ~20% đáp án suy luận thuần không thể self-supervise ở pretrain (không có nhãn) — để finetune lo; hợp lý.

**Trọng số đề xuất khởi đầu:** `gen = read(0.5) + denoise(0.5)`, cả khối gen hệ số 1.0; MLM/ITM/TWC = 0.5 phụ trợ.

## ĐÃ TRIỂN KHAI (2026-07-03) — mode `gen_all`
Cơ chế: pretrain chạy thêm **1 lượt decoder theo đúng đường finetune** (encoder question-only + ảnh
+ OCR-feature) để sinh **read-scene-text** (chuỗi OCR đọc theo thứ tự spotter). Tái dùng nhánh finetune
qua `self.pretrain=False` (giống `.generate()`). `gen` là trục (hệ số 1.0), MLM/ITM/TWC là phụ trợ (0.5).

> **RÀNG BUỘC QUAN TRỌNG (2026-07-03): KHÔNG sửa `models/`.** Toàn bộ logic gen + vision-unfreeze nằm ở
> tầng **training/data** để finetune (dựng lại model từ config) không bị ảnh hưởng. Lượt decoder gen được
> gọi TỪ TRAINER (`_pretrain_gen_loss`) chứ không nhúng vào `model.forward`. Model modules == `main`.

Files đã sửa (chỉ training/data):
- `data/collator.py`: nhánh pretrain, khi mode∈{gen,gen_all} phát thêm `gen_input_ids` (=question-only
  `q_tok`), `gen_attention_mask`, `gen_labels` (=OCR reading string, cap 64 token, pad→-100).
- `training/metrics.py`: `TaskSpecificTrainer._pretrain_gen_loss` chạy lượt finetune-forward (flip
  `pretrain=False`) để lấy `gen_loss`, stamp vào `outputs["gen_loss"]` trong `compute_loss` + `prediction_step`.
  `simple_pretrain_aggregator` báo `loss_gen`.
- `training/pretraine_loss.py`: mode `gen_all/gen`: `total = gen_loss + 0.5*(mlm+itm+twc)`; stamp `loss_gen`;
  metric vector nới lên 9 phần tử (index 8 = loss_gen).
- `training/pretrain.py`: `use_twc/use_ocr_aug` bật cho `gen_all`; **vision unfreeze bằng `requires_grad`**
  (đọc `--vision_unfreeze_last_n`) sau `model.to(DEVICE)`; smoke `_verify_pretrain_batch` chạy gen-forward + check `[GEN]`.
- `configs/arguments.py`: thêm arg `vision_unfreeze_last_n`.
- `run_pipeline.py`: env passthrough `LOSS_ABLATION_MODE`, `VISION_UNFREEZE_LAST_N`.
- `models/*`, `configs/model_config.py`: **KHÔNG đổi** (== main).

Chạy: `STAGE=pretrain LOSS_ABLATION_MODE=gen_all MOCK_TEST=true bash run_all_colab.sh` (smoke 1-batch),
rồi bỏ `MOCK_TEST` để chạy full. Đọc dòng `[GEN] gen_loss finite & > 0` + `loss_gen` trong eval.

**Nuance đã biết (để cải tiến sau):**
- ITM pollute (tráo ảnh) làm ~50% sample có ảnh ≠ OCR/target; target khớp OCR-branch nên gen vẫn học
  đọc từ OCR-feature, chỉ loãng grounding thị giác ở nửa polluted. Tinh chỉnh khả dĩ: mask gen-loss cho
  sample polluted (giống MLM `keep=1-pollute`) — nhưng cẩn thận batch toàn-polluted (mỗi row vẫn có EOS
  nên không NaN).
- **OCR-denoise** (đòn bẩy phụ) CHƯA làm: cần feed OCR nhiễu-riêng vào branch + target sạch; hoãn sang
  increment sau vì pipeline OCR hiện nối [clean;noisy].
- Chi phí gen_all ≈ 2 lượt forward/step (aux + gen).

## Rủi ro / mở
- Chi phí: thêm 1 lượt decoder/step khi pretrain → chậm hơn; có thể **luân phiên tác vụ theo batch**
  (batch chẵn = read, lẻ = denoise) để không tăng gấp đôi lượt decoder.
- Nguồn OCR = swintextspotter (`.npy`: det_features/rec_features/scores/texts/boxes) — có sẵn cục bộ ở
  `datasets/processed/ViTextVQA/ocr/swintextspotter/`; không cần tải drive khi phát triển/kiểm thử.
