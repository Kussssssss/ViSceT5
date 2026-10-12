# QAClip đối chiếu với QA-ViT

> Cập nhật kiến trúc: Visual Search/ConvNeXt đã được gỡ khỏi model. QAClip vẫn
> hỗ trợ kiểm tra attention trực tiếp, nhưng model tổng dùng `output_attentions=False`.
> Các đoạn nói về Visual Search và kiểm chứng VRAM bên dưới mô tả phiên bản trước
> khi bỏ nhánh này; xem [RESEARCH_CODE_MAP.md](RESEARCH_CODE_MAP.md) cho luồng hiện tại.

Source gốc được clone vào `output/code_review/QA-ViT` (không đưa vào Git).
Commit được đối chiếu: `96faee6ef8bb0157fe366d8315f175e8c4dbe6b8`.

Các nguồn đã đọc:

- [clip_qavit.py](https://github.com/amazon-science/QA-ViT/blob/96faee6ef8bb0157fe366d8315f175e8c4dbe6b8/models/clip_qavit.py): adapter, attention, integration schedules.
- [clip_encoder.py](https://github.com/amazon-science/QA-ViT/blob/96faee6ef8bb0157fe366d8315f175e8c4dbe6b8/models/clip_encoder.py): khởi tạo và nạp CLIP.
- [VisualT5.py](https://github.com/amazon-science/QA-ViT/blob/96faee6ef8bb0157fe366d8315f175e8c4dbe6b8/models/VisualT5.py): nguồn instruction và freeze vision.
- [qavit.yaml](https://github.com/amazon-science/QA-ViT/blob/96faee6ef8bb0157fe366d8315f175e8c4dbe6b8/configs/VisualT5_base/qavit.yaml): cấu hình thí nghiệm gốc.

## Phương pháp được giữ

ViSceT5 vẫn dùng hidden states của ViT5 encoder làm instruction. Mỗi block QA-ViT
có một adapter riêng: `nn.LayerNorm → Linear không bias → GELU → Linear không bias`.
Adapter đưa instruction về chiều CLIP. Ảnh đi qua LayerNorm của block rồi được
ghép với instruction để làm keys/values. Visual queries attend chung tới hai nhóm
keys qua một softmax. Question queries không được trả về.

Joint keys/values dùng thứ tự `[visual, question]` như Eq. (3) của paper. Các hàng
attention là visual queries; cột đầu là CLS và patches, cột sau là question tokens.
Visual Search lấy cột câu hỏi từ index `N_visual` (bao gồm CLS), bỏ hàng CLS.
Việc đổi thứ tự không đổi parameter/checkpoint layout. Code bên ngoài đọc attention
cần dùng offset này thay vì coi các cột đầu là câu hỏi như trước đây.

Output attention đi qua:

```text
out_proj(U) + tanh(instruction_proj_gate) × instruction_out_proj(U)
```

Sau đó là residual và CLIP MLP như backbone. Gate bằng 0 chỉ tắt projection bổ sung;
câu hỏi vẫn tác động qua nhánh `out_proj(U)` ngay từ lúc khởi tạo.

Repo gốc thực sự freeze CLIP trong `VisualT5.freeze_vision_weights()`; config gốc
đặt `freeze_clip: True`. ViSceT5 giữ lựa chọn này: chỉ adapter FFN, projection bổ sung
và gate của QAClip được train. Gradient vẫn truyền về instruction encoder ViT5.

## Khởi tạo và checkpoint

| Thành phần | Khởi tạo khi nạp CLIP gốc |
| --- | --- |
| Embedding, Q/K/V, output projection, LayerNorm và MLP CLIP | Trọng số CLIP pretrained |
| Hai Linear của instruction FFN | Random mặc định PyTorch: uniform ±1/√fan_in |
| LayerNorm của instruction FFN | Scale = 1, bias = 0; eps = 1e-5 |
| `instruction_out_proj` | Copy weight và bias từ `out_proj` của cùng block |
| `instruction_proj_gate` | Scalar 0 |

Repo gốc tạo QA-ViT bằng constructor trước, rồi `load_state_dict()` CLIP và gọi
`init_qavit_comps()`. Vì vậy, lỗi thiếu khởi tạo FFN khi dùng trực tiếp loader
`from_pretrained()` của Transformers 4.45.2 không phải lỗi đường nạp chính của repo gốc.

ViSceT5 giữ API Hugging Face và khởi tạo FFN trong `_init_weights()`. `instruction_dim`,
`integration_point`, `freeze_clip` được xử lý trước khi tạo module. Loader tự copy
projection CLIP cho từng weight/bias bị thiếu trong checkpoint; những adapter đã được
huấn luyện không bị ghi đè. Vì vậy `OpenViVQAModel` không gọi lại `init_qavit_comps()`.
Hàm này chỉ được giữ cho trường hợp chủ động reset projection, không dùng khi resume.

Với `low_cpu_mem_usage=True` mà không chỉ định device map, loader dùng placement CPU
tường minh để tránh Transformers 4.45.2 nhận nhầm các tensor meta cùng shape là
trọng số dùng chung rồi bỏ sót adapter cần khởi tạo. Device map do người gọi chỉ định
vẫn được giữ nguyên.

Tên parameter của các module đang dùng được giữ, nên checkpoint ViSceT5 trước refactor
có thể được nạp. Code gốc còn một `instruction_gate` không được dùng trong forward;
ViSceT5 không tạo tham số này. Nạp trực tiếp checkpoint upstream có thể báo key dư đó.
Toàn bộ checkpoint VisualT5 upstream có cấu trúc khác OpenViVQAModel.

## Những điều chỉnh của ViSceT5

1. Mask token padding của câu hỏi trước softmax. Code upstream nhận mask nhưng không
   dùng trong MM attention. Mask phải có đúng shape `[B, L]`; không tự truncate/pad.
2. Trả attention đúng shape `[B, H, N_visual, L_question + N_visual]`. Upstream reshape
   sai số query khi `output_attentions=True`; đường chạy gốc chủ yếu lấy hidden states.
3. Chỉ tính visual queries thay vì tính cả question queries rồi bỏ đi. Giữ cùng phép
   attention, giảm tensor trung gian không được sử dụng.
4. QK, softmax và attention × V tính FP32 với autocast tắt trong vùng đó. Projection
   và phần còn lại của model vẫn theo dtype của model/autocast.
5. Gradient checkpointing truyền instruction tensor trực tiếp vào checkpoint, giữ
   đúng layer khi recompute. Mặc định non-reentrant để hỗ trợ backbone bị freeze.
6. Cho phép instruction rỗng hoặc `text_emb=None`; giữ cách ablation bỏ câu hỏi.
   Gate/projection đã train vẫn hoạt động khi bỏ câu hỏi, như trước refactor.

Module hỗ trợ lại `all`, `early`, `late`, `late2`, `late3`, `sparse` đúng vị trí upstream.
`OpenViVQAModel` tiếp tục chọn `late`; với CLIP 12 block là block 6–11. Thay đổi schedule
ở model tổng còn cần chọn attention của block QA-ViT phù hợp cho Visual Search.

## Phạm vi so với model gốc

Cấu hình upstream dùng CLIP ViT-L/14-336, FLAN-T5, translation MLP và LoRA. ViSceT5
vẫn dùng các backbone, nhánh OCR và Visual Search hiện hành; lấy patch tokens bỏ CLS
để tạo chuỗi đa phương thức. Refactor này phục hồi và sửa module QAClip, không biến
toàn bộ ViSceT5 thành bản tái hiện thí nghiệm VisualT5 của paper.

## Kiểm chứng

Đối chiếu trực tiếp với class trong source upstream đã clone, dùng cùng trọng số,
FP32/eval, CLIP nhỏ 4 block, instruction chiều 24, không padding và gate = 0.3:
cả sáu integration schedules có sai lệch đầu ra tối đa dưới `7.2e-7`, sai lệch
gradient instruction tối đa dưới `8.2e-8`. Hidden states từng block cũng khớp trong
sai số số học. Phép thử này đối chiếu công thức, không đánh giá chất lượng dataset.

`tests/test_qaclip.py` kiểm tra loader với instruction dimension khác 768, vị trí
integration, giữ trọng số CLIP, freeze, copy projection, checkpoint adapter đã train
và checkpoint thiếu một phần, padding invariance, gradient attention, checkpointed
gradients, CPU BF16 autocast và đối chiếu CLIP khi không có instruction/gate bằng 0.

`tests/test_vqa_pipeline.py` kiểm tra tiếp toàn pipeline: 8 tổ hợp ablation,
forward/backward/generation, train/evaluate/resume và checkpoint model tổng.
Những kiểm tra này dùng backbone nhỏ offline; hiệu quả VQA và bộ nhớ/thời gian GPU
cần được đo bằng thí nghiệm trên backbone và dataset thực tế.

## Kiểm tra tham số khi VRAM thay đổi

Với CLIP-B/16, instruction chiều 768, integration `late`, freeze CLIP:

| QAClip | Trước refactor (`d642998`) | Sau refactor (`b5f62ea`) |
| --- | ---: | ---: |
| Tổng tham số | 96,430,086 | 96,430,086 |
| Có thể học | 10,630,662 | 10,630,662 |
| Instruction FFN | 7,087,104 | 7,087,104 |
| Projection bổ sung | 3,543,552 | 3,543,552 |
| Gate | 6 | 6 |

Tên, shape và cờ `requires_grad` của toàn bộ parameter khớp giữa hai bản. Đổi sang
`[visual, question]` cũng giữ các số này. Gradient checkpointing không tự bật;
encoder vẫn có mặc định `gradient_checkpointing=False`. Refactor model tổng ở
`b5f62ea` chỉ bỏ lời gọi copy projection vì loader đã thực hiện việc đó.

Tính visual queries thay vì joint queries đổi shape score từ `(M+K) × (M+K)`
sang `M × (M+K)`. Với M=197 và K=32, số phần tử score giảm khoảng 14%, chỉ tại
các block QA-ViT. Đây không phải phép đo mức giảm VRAM của toàn hệ thống.

Một probe CPU FP32, cùng shape và trọng số, B=4, M=197, K=32, D=64, 4 block
(2 block QA), dùng saved-tensor hooks và đếm storage không trùng, bỏ storage
parameter: `d642998` lưu 10.36 MiB, `b5f62ea` lưu 15.59 MiB cho backward.
Các adapter vẫn nhận gradient ở cả hai bản. Probe này không đo peak CUDA hay
bộ nhớ GPU của pipeline đầy đủ, nhưng không hỗ trợ kết luận rằng graph bị cắt
hoặc rằng riêng tối ưu query giải thích mức giảm từ 32 GB xuống 12 GB.

Để xác định nguyên nhân trên A100, cần đo trong chính process training, cùng một
batch đã collate, sau warm-up và qua optimizer step; so sánh từng bản bằng process
mới. Phân biệt `torch.cuda.memory_allocated()`, `memory_reserved()` và
`max_memory_allocated()`. Reset peak trước đoạn forward/backward/optimizer cần đo.
Nếu training chạy bằng subprocess trong Colab, chạy các phép đo trong subprocess,
không chạy ở process notebook cha. Chỉ số từ nvidia-smi/Colab có thể bao gồm cache
allocator và bộ nhớ của process khác; chưa có các số đo CUDA này để quy nguyên nhân.
Tham khảo [PyTorch memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management).

## Attribution

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

QA-ViT upstream được phát hành theo Apache-2.0. Bản license upstream được giữ tại
[QA_VIT_LICENSE.txt](QA_VIT_LICENSE.txt). Các thay đổi của ViSceT5 được mô tả ở trên
và trong docstring của `models/modules/qa_clip.py`.
