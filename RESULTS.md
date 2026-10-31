# ViSceT5 — Tổng hợp kết quả

Nguồn: 68 repo của `Kus669` trên HuggingFace, đọc ngày 2026-10-28. Số liệu lấy từ các file sau trong mỗi repo:

- `trainer_state.json`: điểm dev theo từng epoch.
- `predictions/**/eval_test.json`: điểm test.
- `config.json` và `training_args.bin`: cấu hình của run.

## Quy ước

- **Dev**: EM / F1 (%) trên tập val, chấm trong lúc train. Cột "best" là epoch có (EM + F1)/2 cao nhất, "ep5" là epoch cuối.
- **Test**: EM / F1 token-level (%) chấm bằng `training/benchmark_eval.py` (giống `benchmark_evaluation.ipynb`, GT ở `eval_gt/`), trên 10.028 câu test ViTextVQA. Checkpoint được chấm ghi trong ngoặc.
- **Batch**: `per_device_batch × grad_accum` (1 GPU). Số step mỗi epoch: 4395 → batch hiệu dụng 8, 2197 → 16, 1099 → 32.
- **Code**: commit dùng để train. Từ `4221b1f`, commit nằm luôn trong tên repo (`ViSceT5-<sha7>-<dataset>-<run>`).
- Dấu `–` nghĩa là repo không có số liệu đó (chưa predict test, hoặc train bị dừng giữa chừng).

---

## 1. Ablation TWC hiện tại (ViTextVQA, QA-ViT off)

Các switch: `USE_CONSFORMER`, `USE_TWC` (p = 0.35, τ0 = 0.2). Với cùng một commit và cùng một batch, các run chỉ khác nhau ở hai switch này.

### 1a. `cc773b0`: mọi run có đủ đặc trưng OCR (chữ + ký tự + box + det + rec)

Ở commit này, bản related lúc test vẫn được tạo **như lúc train** (giữ / nhiễu / sửa).

| Batch, LR | Consformer | TWC | Dev best | Dev ep5 | **Test** | Repo |
|---|:-:|:-:|---|---|---|---|
| 32×1, 1e-4 | – | – | ep5 31.10 / 57.16 | 31.10 / 57.16 | **32.56 / 56.64** | `ViSceT5-cc773b0-vitextvqa-base-bs32x1` |
| 32×1, 1e-4 | ✓ | – | ep4 30.79 / 57.04 | 30.61 / 56.93 | 31.74 / 56.23 | `ViSceT5-cc773b0-vitextvqa-consformer-bs32x1` |
| 32×1, 1e-4 | – | ✓ | ep5 30.94 / 56.76 | 30.94 / 56.76 | 32.13 / 56.06 | `ViSceT5-cc773b0-vitextvqa-twc-p35-tau0.2-bs32x1` |
| 32×1, 1e-4 | ✓ | ✓ | ep5 30.94 / 57.04 | 30.94 / 57.04 | 31.97 / 55.93 | `ViSceT5-cc773b0-vitextvqa-consformer-twc-p35-tau0.2-bs32x1` |
| 4×2, 3e-5 | – | – | ep5 30.73 / 56.76 | 30.73 / 56.76 | 31.53 / 56.01 | `ViSceT5-cc773b0-vitextvqa-base-bs4x2` |
| 4×2, 3e-5 | – | ✓ | ep5 30.44 / 56.46 | 30.44 / 56.46 | 31.47 / 55.74 | `ViSceT5-cc773b0-vitextvqa-twc-p35-tau0.2-bs4x2` |

Test được chấm ở checkpoint cuối: `checkpoint-5495` cho batch 32×1, `checkpoint-21975` cho batch 4×2.

### 1b. `4221b1f`: khi tắt Consformer, OCR chỉ có chữ (text-only)

Commit `4221b1f` là code của run tham chiếu bs4x2.

| Batch, LR | Consformer | TWC | Dev best | Dev ep5 | **Test** | Repo |
|---|:-:|:-:|---|---|---|---|
| 4×2, 3e-5 | – | – | ep4 27.08 / 53.55 | 27.06 / 53.51 | 28.55 / 52.77 | `ViSceT5-4221b1f-vitextvqa-base-bs4x2` |
| 4×2, 3e-5 | – | ✓ | ep5 27.08 / 53.30 | 27.08 / 53.30 | 28.34 / 52.40 | `ViSceT5-4221b1f-vitextvqa-twc-p35-tau0.2-bs4x2` |
| 4×2, 3e-5 | ✓ | ✓ | ep4 29.91 / 56.04 | 30.32 / 56.68¹ | **31.64 / 56.09** | `ViSceT5-twcbr-vitextvqa-consformer-twc-p35-tau0.2-bs4x2` (run tham chiếu) |
| 8×2, 3e-5 | ✓ | ✓ | ep5 29.16 / 55.65 | 29.16 / 55.65 | 29.94 / 54.91 | `ViSceT5-twcbr-vitextvqa-consformer-twc-p35-tau0.2-bs8x2` |

¹ Lần đánh giá val ở epoch 5 của run này bị lỗi khi resume. Số 30.32 / 56.68 là kết quả predict lại `checkpoint-21975` bằng cell 6b.

### 1c. `a04a02e`: bản related lúc val/test/predict được sửa 100%

| Batch, LR | Consformer | TWC | Dev | Test | Repo |
|---|:-:|:-:|---|---|---|
| 32×1, 1e-4 | ✓ | – | đang chạy | – | `ViSceT5-a04a02e-vitextvqa-consformer-bs32x1` (hiện mới có `.gitattributes`) |

Run chỉ có Consformer không dùng bản related, nên ở `a04a02e` nó phải cho kết quả tương đương run 32×1 + Consformer ở `cc773b0`. Hai run cần chạy lại ở `a04a02e` là `twc` và `consformer-twc`.

Run `consformer` ở `a04a02e` vẫn dùng ConsFormer lỗi (mục 6). Mọi run có Consformer cần chạy ở `377099e` trở đi.

### 1d. `377099e`: ConsFormer có prior thành phần đúng (Tree Transformer / ViConsFormer)

| Batch, LR | Consformer | TWC | Dev best | Dev ep5 | **Test** | Repo |
|---|:-:|:-:|---|---|---|---|
| 32×1, 1e-4 | ✓ | – | ep5 31.35 / 57.06 | 31.35 / 57.06 | 32.15 / 56.20 | `ViSceT5-377099e-vitextvqa-consformer-bs32x1` |

So sánh có cặp trên từng câu test (bootstrap CI 95%, McNemar), với các run 32×1 ở `cc773b0`:

| So sánh | ΔEM [CI 95%] | McNemar p |
|---|---|---|
| Consformer đã sửa − baseline | −0.41 [−1.03, +0.21] | 0.21 |
| Consformer đã sửa − Consformer lỗi | +0.41 [−0.17, +1.01] | 0.18 |
| Consformer đã sửa − TWC | +0.02 [−0.57, +0.63] | 0.97 |
| Consformer lỗi − baseline | −0.82 [−1.44, −0.20] | 0.009 |

Hai run bất kỳ trả lời khác nhau ở khoảng 1.000 câu test (trùng nhau khoảng 90% trên EM). Vì vậy với một seed, chênh lệch dưới khoảng ±0.6 EM là nhiễu.

### 1e. `b744ce2`: ConsFormer residual (embedding token ViT5 + đầu ra ConsFormer)

| Batch, LR | Consformer | TWC | Dev ep5 | **Test** | Repo |
|---|:-:|:-:|---|---|---|
| 32×1, 1e-4 | ✓ | – | 31.12 / 57.26 | 31.87 / 56.19 | `ViSceT5-b744ce2-vitextvqa-consformer-bs32x1` |
| 32×1, 1e-4 | ✓ | ✓ (p 0.35, τ 0.07) | 30.22 / 56.01 | chưa predict | `ViSceT5-b744ce2-vitextvqa-consformer-twc-p35-tau0.07-bs32x1` |

### 1f. `0bbd575`: copy head, loss liên kết, chép cụm từ (đã gỡ ở `2ec2fd7`)

| Batch, LR | Consformer | TWC | Dev ep1→5 (EM) | Dev ep5 | **Test** | Repo |
|---|:-:|:-:|---|---|---|---|
| 32×1, 1e-4 | ✓ | ✓ (p 0.35, τ 0.07) | 17.21 · 24.85 · 28.44 · 29.27 · 29.58 | 29.58 / 55.78 | chưa predict | `ViSceT5-0bbd575-vitextvqa-consformer-twc-p35-tau0.07-bs32x1` |

### Nhận xét

- **Đặc trưng OCR đầy đủ là yếu tố lớn nhất.** Baseline text-only đạt 28.55 test EM, còn baseline có ký tự, box, det và rec đạt 31.53. Chênh lệch là +3.0 EM, cùng batch 4×2 và LR 3e-5.
- **Batch 32 với LR 1e-4 tốt hơn batch 8 với LR 3e-5 khoảng +1 EM** (baseline: 32.56 so với 31.53). Batch 16 với LR 3e-5 lại kém hơn batch 8 (29.94 so với 31.64).
- **Consformer của mọi run trên đều lỗi** (mục 6), nên các số "+ Consformer" chưa đo được đóng góp thật của module này.
- **Ở `cc773b0`, chưa module nào vượt baseline trên test.** Thứ tự là base 32.56 > TWC 32.13 > cả hai 31.97 > Consformer 31.74. Các chênh lệch trên dev đều dưới 0.4 EM, nằm trong mức dao động giữa các lần chạy (khoảng 1 EM). Riêng TWC lúc test còn nhận bản related có nhiễu, nên cần đo lại ở `a04a02e`.

---

## 2. Nhánh `exp/twc-ocr-centric` trước refactor `bf74388` (ViTextVQA, tháng 10)

Ký hiệu cấu hình trong tên repo:

- `scespavis` = Consformer.
- `ocraug` = OCR Augmentation.
- `lex` / `ans` = đích của TWC: từ vựng / đáp án.
- `q` = kích thước queue.
- `lat` = lattice.
- `noimg` = không dùng ảnh.
- `vast` = chạy trên máy Vast.

| Repo | Batch | Cấu hình chính | Dev best | Test |
|---|---|---|---|---|
| `ViSceT5-abl-vitextvqa-noimg-scespavis-ocraug-r1` | 8×1 | Consformer + OCR-Aug, không ảnh | ep4 29.23 / 55.26 | 30.77 / 54.84 |
| `ViSceT5-twcbr-vitextvqa-scespavis-ocraug-twc-t1` | 8×1 | + TWC p0.35 | ep5 29.64 / 56.03 | 30.89 / 55.46 |
| `ViSceT5-repro-7b99d60-only_consformer-c1` | 4×2 | tái lập only_consformer (code cũ `7b99d60`) | ep5 29.27 / 55.99 | 31.69 / 55.73 (ckpt-21975, bước 1) · 31.23 / 55.88 (bước 2) |
| `ViSceT5-twcbr-vitextvqa-noimg-scespavis-ocraug-twc-q16384-t5` | 8×2 | TWC, queue 16384, không ảnh, LS 0 | ep5 27.33 / 53.62 | 28.07 / 52.87 |
| `ViSceT5-twcbr-vitextvqa-scespavis-ocraug-twc-ans-t6` | 8×2 | TWC theo đáp án, queue 16384, LS 0 | ep4 28.77 / 55.43 | 29.71 / 54.49 |
| `ViSceT5-twcbr-vitextvqa-scespavis-ocraug-twc-lex-q0-rms-noadapter-t9` | 8×1 | TWC từ vựng p0.7 τ0.4, RMSNorm, không adapter | ep5 29.74 / 56.07 | – |
| `ViSceT5-twcbr-vitextvqa-scespavis-ocraug-twc-lex-q0-rms-noadapter-t11` | 8×1 | p1.0 τ0.1 (dừng ở ep3) | ep3 26.73 / 54.04 | – |
| `ViSceT5-twcbr-vitextvqa-scespavis-vast1` | 4×2 | chỉ Consformer + adapter muộn, LS 0 | ep5 30.38 / 56.26 | 30.98 / 55.43 |
| `ViSceT5-twcbr-vitextvqa-scespavis-ocraug-twc-lex-q0` | 8×1 | TWC từ vựng p0.6 τ0.1 + adapter muộn | ep5 30.36 / 56.43 | 30.93 / 55.41 |
| `ViSceT5-twcbr-vitextvqa-scespavis-ocraug-lat-p70-twc-lex-q0` | 8×1 | OCR-Aug lattice p0.7 + TWC từ vựng | ep5 29.85 / 56.51 | 31.38 / 55.95 |
| `tmp_viscet5` (nhánh `tmp`) | 8×1 | OCR-Aug separate p0.6, τ0.07 cố định, không adapter | ep5 **31.10 / 57.44** | 31.59 / 55.75 |

`ViSceT5-twcbr-vitextvqa-scespavis-t1` và `-t4` là repo rỗng (run không xong).

---

## 3. Ablation module cũ (`exp/ablation-drop-ocr`, tháng 8, ViTextVQA)

Các run này chỉ có dev, chưa run nào được predict test trên HF. Cấu hình chung: 4×2, LR 3e-5, 5 epoch.

| Repo | QA-CLIP | VS/AVF | Consformer | OCR-Aug | Dev best | Chọn ckpt theo |
|---|:-:|:-:|:-:|:-:|---|---|
| `ViSceT5-ft-vitextvqa-baseline-b4` | – | – | – | – | ep5 29.04 / 55.41 | em |
| `ViSceT5-ft-vitextvqa-qaclip-b4` | ✓ | – | – | – | ep5 29.58 / 56.03 | em |
| `ViSceT5-ft-vitextvqa-qaclip-b6` | ✓ | – | – | – | ep5 29.78 / 56.06 | em |
| `ViSceT5-ft-vitextvqa-avf-b4` | – | ✓ | – | – | ep5 29.35 / 55.40 | em |
| `ViSceT5-ft-vitextvqa-aug-b4` | – | – | – | ✓ | ep5 **30.24 / 56.47** | em |
| `ViSceT5-ft-vitextvqa-only_consformer` | – | – | ✓ | – | ep5 29.91 / 56.08 | loss |
| `ViSceT5-ft-vitextvqa-only_qaclip-all` | (tất cả) | | | | ep4 29.54 / 55.96 | loss |
| `ViSceT5-ft-vitextvqa-full-noaug` | ✓ | ✓ | ✓ | – | ep5 29.39 / 56.01 | loss |
| `ViSceT5-ft-vitextvqa-full` | ✓ | ✓ | ✓ | ✓ | ep5 29.85 / 55.92 | loss |
| `ViSceT5-ft-vitextvqa-only_qaclip` / `-only_vs` / `-only_ocr_aug` | | | | | ep5 23.69 / 23.84 / 23.47 | loss |
| `ViSceT5-ft-vitextvqa-only_qaclip-ver2` | ✓ | – | – | – | ep5 25.96 / 52.08 | loss |
| `ViSceT5-noocr-baseline` / `-noocr-qaclip` | không OCR | | | | ep3 4.21 / 4.52 | em |

Ba điều cần lưu ý khi đọc bảng này:

- Ba run `only_qaclip`, `only_vs`, `only_ocr_aug` (khoảng 23.5 EM) thấp hơn hẳn các run `-b4` có cùng cấu hình (khoảng 29). Chúng chạy trước các bản sửa lỗi ở mục 6; nguyên nhân cụ thể chưa được kiểm tra.
- `ViSceT5-ft-openvivqa-only_consformer` có số dev **trùng hoàn toàn** với `ViSceT5-ft-vitextvqa-only_ocr_aug`, nên nhiều khả năng đã upload nhầm. Không nên dùng repo này.
- `ViSceT5-ft-vitextvqa-baseline` chỉ có 2 epoch.

---

## 4. Pretrain → finetune và các dataset khác (tháng 7–8)

| Repo | Dataset | Ghi chú | Dev best |
|---|---|---|---|
| `ViSceT5-finetune-frompretrain` | ViTextVQA | QA-CLIP + VS + Consformer, từ pretrain | ep5 28.94 / 55.61 |
| `ViSceT5-finetune-frompretrain-8400` | ViTextVQA | từ pretrain-version2 | ep5 29.08 / 56.03 |
| `ViSceT5-finetune-frompretrain-v4-10985` | ViTextVQA | từ pretrain-genall-v4 | ep5 29.25 / 55.92 |
| `ViSceT5-finetune-vitextvqa-scratch` | ViTextVQA | không pretrain (dừng ở ep2) | ep2 24.85 / 51.23 |
| `ViSceT5-finetune-openvivqa-scratch-branch` | ViTextVQA² | không pretrain | ep4 29.51 / 56.05 |
| `ViSceT5-finetune-viocrvqa-frompretrain` | ViOCRVQA | từ pretrain | ep5 54.81 / 75.15 |
| `ViSceT5-finetune-viocrvqa-pretrained` | ViOCRVQA | dừng ở ep3 | ep3 53.26 / 73.94 |
| `ViSceT5-finetune-openvivqa-scratch` | OpenViVQA | | ep4 13.12 / 57.96 |
| `ViSceT5-ft-openvivqa-baseline-b4` | OpenViVQA | | ep4 12.92 / 58.47 |
| `ViSceT5-ft-openvivqa-baseline` | OpenViVQA | 2 epoch | ep2 11.14 / 55.56 |

² Tên repo ghi `openvivqa`, nhưng checkpoint tốt nhất là `checkpoint-13185` (4395 step mỗi epoch, đúng với ViTextVQA) và điểm dev ở mức ViTextVQA. Run này nhiều khả năng là ViTextVQA.

Test OpenViVQA (chấm offline theo id trên 3.536 câu): CIDEr 3.5581, EM 12.36, F1 57.46. Chi tiết ở `docs/openvivqa_test_eval.md`.

Phiên bản paper cũ (đã comment trong `paper/sections/5_experiments_and_results.tex`) ghi test ViTextVQA như sau:

- ViSceT5: 30.26 / 55.90.
- ViSceT5 có pretrain: 30.38 / 55.74.

**Pretrain** (`ViSceT5-pretrain*`, `-pretrain-genall*`, `-mra-pretrain-ver2/3/4`) chỉ có loss và độ chính xác pretext, không có điểm VQA. Các run pretrain trước khi sửa lỗi dataset đã chạy trên ViTextVQA thay vì VinText + EVJVQA (mục 6).

---

## 5. Bản đồ version (commit → thay đổi)

| Commit | Nhánh | Thay đổi | Run tiêu biểu |
|---|---|---|---|
| `7b99d60` | cũ | code của run only_consformer gốc | `repro-7b99d60-only_consformer-c1` |
| `70e4bd2` / `cc4de3a` | twc-ocr-centric | lattice OCR-Aug, TWC trên output encoder | `...-lat-p70-twc-lex-q0` |
| `bf74388` | twc-ocr-centric | TWC đúng như TWA, bỏ các switch phương pháp | – |
| **`4221b1f`** | twc-ocr-centric | bố cục TWA, chỉ còn 3 switch (QA-ViT, Consformer, TWC) | `...-consformer-twc-p35-tau0.2-bs4x2` (31.64) |
| `5545dce` → `1046e86` | twc-ocr-centric | bản related không có box (TWA `con_remove_bbox`), sau đó bỏ thêm det/rec | – |
| `57e4dc4` → `37ea1a8` | twc-ocr-centric | lấy phần của `tmp`: τ cố định, nhãn kế thừa, bản related sinh theo hash; QA-ViT mặc định off | – |
| **`cc773b0`** | twc-bs4x2-ablation | `4221b1f` + đặc trưng OCR đầy đủ cho mọi ablation | `ViSceT5-cc773b0-*` |
| `6f990d9` | twc-bs4x2-ablation | LR mặc định 1e-4, batch 32×1 | `ViSceT5-cc773b0-*-bs32x1`³ |
| **`a04a02e`** | twc-bs4x2-ablation | bản related lúc val/test/predict được sửa 100% | `ViSceT5-a04a02e-*` |
| **`377099e`** | twc-bs4x2-ablation | ConsFormer có prior thành phần đúng (Tree Transformer / ViConsFormer) + output projection | `ViSceT5-377099e-*` |
| **`b744ce2`** | twc-bs4x2-ablation | ConsFormer residual: đầu ra cộng vào embedding token thay vì thay thế | `ViSceT5-b744ce2-*` |
| **`37a73f4`** | twc-bs4x2-ablation | TWC có copy từ OCR; Consformer có loss liên kết theo đáp án; bật cả hai thì liên kết nâng điểm chép cụm từ | `ViSceT5-37a73f4-*` |
| **`1e5b8ef`** | twc-bs4x2-ablation | Gỡ copy head; TWC hai lượt encoder, bản related chỉ là tăng cường lúc train | `ViSceT5-1e5b8ef-*` |

³ Các run bs32x1 vẫn chạy code `cc773b0`; chỉ notebook đặt LR 1e-4 và batch 32.

## 6. Lỗi đã biết ảnh hưởng tới kết quả cũ

- **ConsFormer: prior thành phần ≈ 0 (có từ commit đầu tiên, kế thừa từ notebook; sửa ở `377099e`).**
  - Code cũ cộng log của **mọi** ô neighbour-attention, không chỉ a[k, k+1], và giữ cả nửa dưới ma trận. Vì vậy C ≤ 0.01 ở mọi ô ngoài đường chéo, trong khi bản tham chiếu cho tới khoảng 0.8.
  - Self-attention có cổng chỉ còn khoảng 0.2% khối lượng (bản tham chiếu: 17–45%). ConsFormer cũ thực chất là 3 lớp FFN chạy trên từng token riêng lẻ.
  - Mọi số "+ Consformer" / `scespavis` / `only_consformer` trước `377099e` đều bị ảnh hưởng.

- **OCR-Aug "phantom pad"**: `<pad>` bị đọc thành "ipad" trong nửa related. Các số OCR-Aug trước khi sửa lỗi này đều bị ảnh hưởng.
- **AVF / QA-CLIP**: score của patch luôn là hằng số, nên AVF crop toàn ảnh. Các run AVF/VS trước khi sửa đều bị ảnh hưởng.
- **DataParallel trên T4 x2**: làm hỏng `pil_images` / `ocr_info` khi scatter. Các run finetune 2 GPU cũ không đáng tin.
- **Label smoothing**: code cũ / nhánh ablation bỏ qua LS 0.1 (dùng CE thuần), còn nhánh TWC có áp dụng. Vì vậy chênh lệch test dưới 1 EM giữa hai nhóm này có thể chỉ là dao động giữa các lần chạy.
- **Pretrain sai dataset**: chạy trên ViTextVQA (là dataset downstream). Mọi kết quả pretrain → finetune cũ không đáng tin.
- **Chọn checkpoint**: tiêu chí `metric_for_best_model` khác nhau theo thời kỳ:
  - `loss`: tháng 7–8;
  - `em`: các run `-b4`;
  - `em_f1`: từ tháng 10.

  Từ `3a2bb7d`, best = trung bình EM và F1 trên val.

## 7. Cách cập nhật

Mỗi run mới đặt tên `ViSceT5-<sha7>-<dataset>-<run>`. Sau khi predict test, thêm một dòng vào mục 1 của file này. Ghi kèm:

- commit;
- batch và LR;
- các switch;
- dev best và dev ep5;
- test, kèm checkpoint đã chấm.
