# Pretrain & Transfer Learning — ghi chú thiết kế ViSceT5

> Mục đích: để các version sau HIỂU pretrain hiện tại sai/thiếu ở đâu, và phát triển đúng
> bản chất transfer learning. Xem sơ đồ kiến trúc: [`pretrain_architecture_gap.svg`](pretrain_architecture_gap.svg).
> Chi tiết mode `gen_all` đã ship: [`../pretrain_decoder_plan.md`](../pretrain_decoder_plan.md).

---

## 1. Khái niệm — phân biệt cho rõ

- **Transfer learning**: tái dùng tri thức/trọng số học được từ tác vụ/miền này cho tác vụ khác.
- **Pretraining**: huấn luyện (thường tự-giám-sát, dữ liệu lớn) để học **ĐẶC TRƯNG tổng quát** *trước* tác vụ đích. Pretrain → finetune là hiện thân phổ biến nhất của transfer learning.
- **Bản chất pretrain = học đặc trưng**, KHÔNG bắt buộc trùng tác vụ finetune. Task-match chỉ là *một* chiến thuật (hữu ích để warm-up decoder), không phải mục tiêu.

**ViSceT5 hiện đang là transfer learning ở 2 tầng:**
1. Dùng **ViT5 (VietAI)** + **CLIP** đã pretrain sẵn = transfer từ mô hình nền (đây đã là transfer learning rồi).
2. Stage "pretrain" của ta trên ViTextVQA = **continued / intermediate / domain-adaptive pretraining** (DAPT/TAPT — Gururangan et al., 2020, *Don't Stop Pretraining*) chồng lên (1).

⇒ Mục tiêu của (2) phải là **làm giàu đặc trưng đa mô thức** (hiểu OCR ↔ ảnh ↔ câu hỏi tiếng Việt) để finetune tốt hơn — chứ không phải bắt chước tác vụ QA.

---

## 2. Pretrain cũ SAI/THIẾU ở đâu (bám sơ đồ)

| # | Vấn đề | Bản chất |
|---|---|---|
| 1 | **Decoder luyện SAI việc** | Cùng khối decoder ViT5 CÓ chạy trong MLM (không phải lớp riêng) → có gradient. NHƯNG MLM = điền chỗ trống teacher-forced (phần lớn copy được) ≠ sinh đáp án tự hồi quy. answer labels = −100. → transfer yếu. |
| 2 | **z_ocr do TWC luyện không có bộ tiêu thụ** | enc_out được decoder cross-attend (được dùng làm ngữ cảnh), nhưng KHÔNG có head chọn đáp án từ z_ocr như pointer của TWA (Eq.8). Độ phân biệt token mà TWC bơm vào gần như vô dụng cho seq2seq. |
| 3 | **Thị giác đóng băng** (`freeze_clip=True`) | Mọi mục tiêu vision (và cả grounding) không chạm được đặc trưng thị giác → pretrain không cải thiện "nhìn". |
| 4 | **Input lệch pha** | Pretrain encoder thấy `question + OCR-text`; finetune thấy `question-only`. |
| 5 | **Head ITM/TWC bị vứt** ở finetune | Tín hiệu học được không có đường chảy tiếp. |

**Điểm mấu chốt (bám TWA):** TWA transfer tốt vì *một trunk dùng chung* được pretrain rồi finetune gắn *head mỏng* (pointer Eq.8 + char-match Eq.9-11) đọc thẳng đặc trưng đó. ViSceT5 tách encoder/decoder; phần quyết định (decoder sinh) không được luyện đúng việc.

---

## 3. Tận dụng TWA/TWC ĐÚNG cách cho tiếng Việt

TWC (sửa lỗi OCR bằng contrastive + Levenshtein) **rất hợp tiếng Việt** vì lỗi OCR tiếng Việt chủ yếu ở **DẤU (diacritics)** và ký tự dễ nhầm — nhưng ta chưa tận dụng hết:

1. **Việt-hoá noise model** (thay augmentation ký tự kiểu English của TWA):
   - bỏ/đổi/thêm **dấu** (à↔á↔ả↔ã↔ạ), telex/VNI (aa→â, ow→ơ), chuẩn hoá NFC/NFD;
   - nhầm ký tự OCR phổ biến: ơ/o, ư/u, đ/d, ê/e, nhầm dấu-mũ;
   - từ điển **tiếng Việt** cho bước "find similar word" (hiện dùng CharBERT English-ish).
2. **Để TWC thực sự TRANSFER trong seq2seq** — biến "sửa lỗi" thành mục tiêu **SINH**:
   - **OCR-denoise generation**: cho OCR **nhiễu** vào, decoder sinh OCR **sạch** → đặt năng lực bền-lỗi vào **decoder** (bộ tiêu thụ), đúng linh hồn TWA nhưng hợp kiến trúc ta.
   - TWC (contrastive, cho encoder) **+** OCR-denoise (generative, cho decoder) = cách seq2seq-đúng để dùng TWA.

---

## 4. Khảo sát phương pháp pretrain (map vào kiến trúc ta)

Chia theo *thứ được luyện*. Mục tiêu: học đặc trưng tốt, không cần trùng QA.

**A. Alignment / contrastive (làm giàu đặc trưng, cho encoder + vision)**
- **ITC** (image-text contrastive, kiểu CLIP/ALBEF/BLIP): căn ảnh↔text toàn cục. Ta có QA-CLIP nhưng đang đóng băng → cần mở băng để ITC có tác dụng.
- **Fine-grained alignment** (FILIP/GLIP-style): căn **OCR-token ↔ vùng ảnh** (dùng bbox) — thay/bổ sung ITM thô.
- **TWC** (token↔word): giữ, Việt-hoá (mục 3).

**B. Masked modeling**
- **MLM** (đang có).
- **MIM / masked-patch** cho vision (nếu mở băng) — học đặc trưng thị giác tự-giám-sát.
- **Masked-region / masked-OCR** có điều kiện ảnh (ép grounding: che token OCR ở CẢ hai nhánh, buộc khôi phục từ ảnh).

**C. Layout-aware (ta CÓ bbox → rất hợp)**
- **LayoutLM / LayoutLMv3**: MLM + masked-image + **word-patch alignment** với toạ độ. Tận dụng `boxes` sẵn có.

**D. Generative / denoising (hợp seq2seq + warm decoder)**
- **Prefix-LM / image-conditioned LM** (SimVLM): sinh text có điều kiện ảnh — học đặc trưng *và* warm decoder.
- **T5 span-denoising có điều kiện ảnh/OCR** (đúng "gốc gác" ViT5).
- **Screenshot/scene-text parsing** (Pix2Struct), **OCR transcription** (TrOCR, Donut) ≈ read-scene-text (mode `gen_all` hiện tại) nhưng có cơ sở lý thuyết vững — xác nhận hướng đi, gợi ý phiên bản *masked* thay vì đọc-toàn-bộ.

**Bài học cân bằng:** `gen_all` lo nhóm **D** (warm decoder). Còn thiếu **A + B + C** (học đặc trưng encoder/vision) — và phải **mở băng vision** thì A/B mới chạm được thị giác.

---

## 5. Lộ trình đề xuất (ưu tiên theo tác động × bản chất transfer)

1. **Mở băng một phần vision** (vài lớp cuối CLIP-vision) — điều kiện cần để mọi mục tiêu vision/grounding có ý nghĩa. ✅ **ĐÃ TRIỂN KHAI** (nhánh `exp/pretrain-gen-all`): arg/env `vision_unfreeze_last_n` / `VISION_UNFREEZE_LAST_N`, áp dụng bằng `requires_grad` **trong `training/pretrain.py`** (KHÔNG sửa `models/`) → mở băng N block vision cuối + `post_layernorm` chỉ trong pretrain. n=2 ≈ +14.2M params; n=0 = giữ đóng băng (mặc định). Vì làm ở tầng training nên finetune (dựng lại model) hoàn toàn không bị ảnh hưởng.
2. **TWC Việt-hoá + OCR-denoise generative** — đưa error-tolerance vào cả encoder (TWC) lẫn decoder (denoise). *(Việt-hoá/denoise: đã có xử lý tiếng Việt + denoise một phần — chưa sửa lại ở đợt này theo yêu cầu.)*
3. **Fine-grained OCR↔region alignment** (dùng bbox) — thay ITM thô.
4. **Layout-aware objective** — tận dụng toạ độ.
5. **Cân đối trọng số loss**; giữ `gen_all` là warm-up decoder, không phải trục duy nhất.
6. **Đánh giá đúng "chất lượng đặc trưng"**, không chỉ loss (mục 6).

---

## 6. Đo "đặc trưng tốt" — đúng tinh thần transfer learning

- **Linear/attentive probing**: đóng băng encoder sau pretrain, gắn head tuyến tính cho tác vụ phụ (ảnh có chữ? đáp án nằm trong OCR? OCR↔image retrieval) → đo đặc trưng có giàu lên không.
- **from-pretrain vs from-scratch** cùng seed/step (bằng chứng transfer thực).
- **Zero-shot OCR↔image retrieval** trước/sau pretrain.
- Nếu probing không cải thiện dù loss pretrain đẹp → mục tiêu pretrain đang học "tắt" (trivial), cần xem lại (xem [`../twc_leak_analysis.md`](../twc_leak_analysis.md)).

---

## 7. TL;DR cho version sau

- Ta đang làm transfer learning 2 tầng; stage "pretrain" của ta là *intermediate pretraining* → mục tiêu là **đặc trưng**, không phải QA.
- Sai cũ: decoder luyện **sai việc** (MLM), z_ocr(TWC) **không có bộ tiêu thụ**, **vision đóng băng**, input **lệch**.
- `gen_all` mới chỉ **warm-up decoder** (nhóm generative). Bước tiếp: **mở băng vision + alignment/contrastive Việt-hoá + layout-aware**, và **đo bằng probing** chứ không chỉ nhìn loss.
