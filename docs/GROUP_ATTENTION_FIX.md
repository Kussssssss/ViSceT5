# Sửa GroupAttention theo code gốc được cung cấp

Nguồn đối chiếu: các lớp `GroupAttention`, `OCREncoderLayer`, `OCREncoder` và
`ScaledDotProductAttention` trong source người dùng gửi ngày 28/10/2026.

## Ba lỗi đã sửa

| Vị trí | Trước sửa | Sau sửa, theo source gốc |
| --- | --- | --- |
| Score của GroupAttention | `Q @ K.T / sqrt(d_k)` | `Q @ K.T / d_k` |
| Log trước khi tích lũy đường đi | Lấy log toàn bộ ma trận neighbor affinity | Chỉ giữ `log(neighbor[k,k+1] + 1e-9)` trên đường chéo kề trên; vị trí khác bằng 0 |
| Ma trận group trước khi đối xứng | Giữ cả đường chéo và tam giác dưới sau `exp` | Chỉ giữ tam giác trên nghiêm ngặt, rồi cộng bản chuyển vị và thành phần diagonal/epsilon như gốc |

Thay đổi đầu tiên chỉ áp dụng cho GroupAttention. Self-attention Q/K/V thông thường
vẫn dùng `sqrt(d_kv)`, đúng như source gốc.

## Công thức sau sửa

Với mỗi head, normalize context rồi chiếu query/key như trước:

```text
P = softmax(mask_adjacent(Q @ K.T / d_k))
A = sqrt(P * P.T + 1e-4)
N = prior + (1 - prior) * A
```

`prior` là **group matrix của lớp trước**, không phải neighbor matrix và không
detach. Lớp đầu bắt đầu với prior = 0. Luồng này đã đúng trước khi sửa và được giữ.

Với i < j, group affinity tích lũy qua các liên kết liền kề:

```text
G[i,j] = product(N[k,k+1] + 1e-9, k=i..j-1) + 1e-4
G[j,i] = G[i,j]
G[i,i] = N[i,i]
```

Ví dụ hai liên kết có affinity 0.8 và 0.6 thì affinity qua cả hai xấp xỉ 0.4801.
Giá trị ở các ô không liền kề của N không được đưa vào tích đường đi.

Code dùng phép nhân ma trận tam giác để tính đồng thời các tích trong log space:

```python
adjacent_log = torch.log(neibor_attn + 1e-9).masked_fill(~forward_neighbor, 0)
t = adjacent_log.matmul(tri_f)
g_attn = tri_f.matmul(t).exp().masked_fill(~(tri & ~eye), 0)
g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(~eye, 1e-4)
```

Trước sửa, log của các ô ngoài đường chéo kề trên tham gia phép nhân, làm group
affinity phụ thuộc vào các ô không thuộc đường đi. Đồng thời, không xóa tam giác
dưới/diagonal trước khi đối xứng khiến các phần thừa sau `exp` bị cộng vào G.
Ngay trường hợp một token và prior=0, diagonal cũ xấp xỉ 3.00015, trong khi source
gốc trả về xấp xỉ 1.00005. Không ép diagonal về 1 vì đó sẽ là một thay đổi khác
với công thức source được cung cấp.

## Những phần giữ nguyên và khác biệt có chủ đích với snippet

- Normalize, các projection query/key, residual, FFN và ba lớp OCR encoder giữ nguyên.
- GroupAttention nhân vào trọng số self-attention **sau softmax**; không thêm bước
  normalize lại, không đổi thành additive attention bias.
- Neighbor affinity được tính đối xứng bằng geometric mean và giữ nguyên hai
  epsilon 1e-4 / 1e-9 của source gốc.
- Tính score, log và exp ở FP32; trả về dtype của context như code hiện hành.
- API dùng mask token B×L, tạo neighbor mask B×L×L hợp lệ cho cả query/key; riêng
  self-attention broadcast key mask thành B×1×1×L. Không đổi collator/mapping OCR.
- Snippet gốc gọi `att.masked_fill(...)` nhưng không gán kết quả và không dùng bản
  in-place. Dòng đó không áp dụng mask. Giữ phép gán `scores = scores.masked_fill(...)`
  đang đúng trong repository để padding không góp vào keys/values.
- Source gốc khai báo dropout của GroupAttention nhưng không gọi nó trong forward;
  không thêm dropout mới vào công thức.

Không thêm/xóa parameter, không đổi tên/shape state-dict keys. Checkpoint cũ vẫn
nạp được, nhưng output khi bật ConsFormer thay đổi do sửa công thức. Điểm số trước
và sau sửa cần được đo bằng thực nghiệm, không suy ra từ tests.

## Kiểm chứng

`tests/test_ocr_consformer.py` chứa phép tính tham chiếu độc lập từ công thức gốc:

- So output và gradient theo context/projection/norm trên độ dài 1, 2, 5, 9,
  với prior=0 và prior tensor.
- Kiểm tra tích các liên kết kề, tính đối xứng và diagonal; đổi prior ngoài đường
  đi không làm đổi group matrix.
- So output OCR encoder ba lớp và neighbor matrices từng lớp với công thức gốc.
- Kiểm tra thêm padding không đổi output của các token thật trong chuỗi có ít
  nhất một liên kết kề, và self-attention không đọc key/value đã mask.
- Kiểm tra output/backward hữu hạn ở FP32, FP16, BF16, chuỗi một token và mask
  toàn padding; kiểm tra self-attention vẫn dùng sqrt(d_kv).

Các tests này chạy trên CPU, không đo hiệu suất VQA hoặc VRAM GPU.

## Cấu hình thí nghiệm hiện tại

YAML và notebook bật ConsFormer để dùng GroupAttention đã sửa, vẫn tắt augmentation/TWC.
Các cờ hiện tại là `ablation_use_consformer=true` (YAML), `USE_CONSFORMER=True`
(notebook), với `ablation_use_ocr=true` / `USE_SCESPAVIS=True`.
Các thông số learning rate/batch size và công thức loss không đổi trong lần sửa này.
