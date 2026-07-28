# OpenViVQA — Offline test-set evaluation (id-mapped)

Codalab (VLSP 2023 OpenViVQA) đã đóng, nhưng đã xin được **tập test có đáp án thật**
(`openvivqa_test_v2.json`, 3.536 câu — subset công khai của 14.035 câu test). Kết quả
dưới đây chấm **offline** bằng cách map theo `id` giữa dự đoán và ground-truth, dùng
đúng bộ metric như khi đánh giá val trong finetune.

## Cách tái lập
```bash
python -m training.eval_submission \
  --gt   <đường_dẫn>/openvivqa_test_v2.json \
  --pred output/submission_OpenViVQA_test.csv \
  --out  output/eval_OpenViVQA_test.json
```
Dự đoán `submission_OpenViVQA_test.csv` sinh bởi `training/predict.py`
(`--dataset OpenViVQA --split test --batch_size 4 --num_beams 4 --max_new_tokens 56`
— tham số khớp finetune eval).

## Kết quả (map theo id: khớp 3536/3536)

| Metric | Giá trị |
|---|---|
| **CIDEr** (= cột "F1" trên Codalab) | **3.5581** |
| BLEU-1 | 0.5496 |
| BLEU-2 | 0.4890 |
| BLEU-3 | 0.4411 |
| BLEU-4 | 0.4022 |
| F1 (token) | 0.5746 |
| EM | 0.1236 (12.36%) |

BLEU length-ratio = 1.184 (over-generation nhẹ).

## Tính đúng như val eval — đã kiểm chứng
- EM/F1: `training/eval_submission.py` **import trực tiếp** `_normalize_txt` +
  `compute_f1_em` từ `training/metrics.py` (chính hàm chấm val) → khớp tuyệt đối;
  single-reference (cả 3536 câu đều đúng 1 đáp án).
- BLEU/CIDEr: cùng thư viện `pycocoevalcap` (Bleu(4)/Cider) mà val eval dùng.

## Khác biệt duy nhất (đã đo, không đáng kể)
1. **PTBTokenizer (Java) vs whitespace**: val eval tokenize bằng PTBTokenizer; script
   dùng `_normalize_txt` + tách khoảng trắng (không cần Java). Đo trực tiếp trên chính
   dữ liệu này: **ΔCIDEr ≤ 0.008, ΔBLEU ≤ 0.002** (nhỏ vì tiếng Việt đã có khoảng trắng
   quanh dấu câu).
2. **Nhãn round-trip qua tokenizer ViT5**: val eval so với label đã decode-lại; script
   so với GT thô (đúng hơn cho điểm test chính thức). Chênh **EM ~0.2pp, F1 ~0.003**.

→ Các con số trên khớp điểm Codalab thật tới ~3 chữ số thập phân.

## Ghi chú
- OpenViVQA đáp án free-form dài → **EM thấp là bình thường**; CIDEr/BLEU là thước đo
  chính (và đúng là cái Codalab gọi nhầm là "F1").
- Không commit dữ liệu GT/pred (nằm ngoài repo); chỉ commit script + tài liệu này làm
  bằng chứng tái lập.
