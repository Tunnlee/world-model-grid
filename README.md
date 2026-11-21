# World Model Grid

Môi trường grid 8×8 nhỏ để học và thử nghiệm world model cho bài toán học tăng cường. Mục tiêu cuối cùng là giúp agent sống sót lâu nhất có thể.

## 1. Nguồn cảm hứng

Ý tưởng được gợi cảm hứng từ:

> David Ha and Jürgen Schmidhuber. “World Models.” *arXiv preprint* arXiv:1803.10122 (2018). https://doi.org/10.48550/arXiv.1803.10122

- [Bài báo trên arXiv](https://arxiv.org/abs/1803.10122)
- [Bản trình bày tương tác](https://worldmodels.github.io/)

Đây là dự án học tập độc lập; code hiện tại không tái tạo đầy đủ kiến trúc hay kết quả trong bài báo.

## 2. Trạng thái dự án

- **Môi trường:** `GridEnv8x8` tạo lưới 8×8 với 5–10 vật cản tĩnh và một agent. Agent có bốn hành động lên, xuống, trái, phải. Mỗi bước an toàn được +1 điểm; va vào đá hoặc ra ngoài biên nhận -10 và kết thúc episode (`terminated=True`). Episode sống đến 100 bước dừng do giới hạn (`truncated=True`).
- **Observation:** mảng `float32` shape `(3, 8, 8)`, one-hot theo channel-first: ô trống (0), vật cản (1), agent (2).
- **Model V:** MLP autoencoder đã có thể mã hóa và tái tạo observation. Hiện model chưa dự đoán state tiếp theo theo action.
- **Các phần tiếp theo:** `model_m.py`, `dream_train.py` và `evaluate.py` hiện chưa được triển khai.

## 3. Cài đặt

Cần Python 3.14 trở lên và [uv](https://docs.astral.sh/uv/). Từ thư mục gốc repo:

```bash
uv sync
```

## 4. Chạy test

```bash
uv run pytest
```

## 5. Thu thập dữ liệu

Collector tạo 100.000 episode train hoặc 10.000 episode validation. Trong mỗi split, 70% episode dùng policy ngẫu nhiên; 30% dùng random walk chọn hướng không va chạm ngay ở bước kế tiếp.

```bash
uv run python -m world_model_grid.collect_data train
uv run python -m world_model_grid.collect_data val
```

Dataset được lưu tại `data/train_dataset.npz` và `data/val_dataset.npz`. Thư mục `data/` không được commit lên Git. Các episode dài ngắn khác nhau nên được lưu dưới dạng mảng NumPy `dtype=object`; khi nạp cần bật pickle:

```python
import numpy as np

with np.load("data/train_dataset.npz", allow_pickle=True) as data:
    states = data["states"]
```

Dataset lưu `states`, `actions`, `rewards`, `next_states` và `terminated`. Collector dùng `truncated` để dừng episode nhưng không lưu cờ này. Toàn bộ split được giữ trong RAM trước khi ghi file.

## 6. Chạy Model V

Từ thư mục gốc repo, chạy file trực tiếp với một trong các mode mà `sys.argv[1]` hỗ trợ:

**Train Model V** — cần có cả dataset train và validation. Lệnh log kết quả lên Weights & Biases và lưu checkpoint tốt nhất theo validation loss tại `weights/model_v_parameters.pth`.

```bash
uv run python src/world_model_grid/model_v.py train
```

**Tính metrics trên validation set** — cần checkpoint đã được train.

```bash
uv run python src/world_model_grid/model_v.py metrics
```

**Lưu ảnh so sánh reconstruction** — cần checkpoint và validation dataset; ảnh được lưu tại `results/model_v_recon.png`.

```bash
uv run python src/world_model_grid/model_v.py recon
```

CLI hiện nhận `train`, `metrics` và `recon`; `val` không phải mode dòng lệnh (chỉ là tên split được dùng bên trong code). Hãy truyền đúng một mode sau tên file.

## 7. Lựa chọn thiết kế

| Lựa chọn | Lý do | Trạng thái |
| --- | --- | --- |
| CrossEntropy theo từng ô | Mỗi ô thuộc một trong ba lớp rời rạc: trống, vật cản hoặc agent. Decoder xuất logits cho ba lớp tại mỗi ô, nên CrossEntropy phù hợp với phân loại đa lớp theo ô. | Đang dùng trong train và validation. |
| Class weight nghịch đảo tần suất | Lớp hiếm chịu loss lớn hơn để giảm bỏ sót vật cản. Trọng số được tính từ tần suất nhãn trong từng batch nên thay đổi theo batch; cách này có thể tăng dự đoán đá giả. | Đang dùng; trade-off được so sánh ở mục 8. |
| Latent dimension = 16 | Nén observation 192 giá trị xuống vector 16 chiều, qua hidden layer 64. | Kích thước thử nghiệm ban đầu; chưa có so sánh để kết luận là tối ưu. |
| Adam, learning rate 0.0015, batch size 256, 50 epochs | Cấu hình baseline hiện tại; checkpoint được chọn theo validation loss. | Cần tiếp tục đánh giá cùng các cấu hình khác. |

Model V là autoencoder xác định, chưa phải VAE và chưa được huấn luyện để dự đoán dynamics.

## 8. Tóm tắt thí nghiệm class weight

| Runs | Cách đặt trọng số đá | Kết quả quan sát |
| --- | --- | --- |
| 3–4 | Giữ nguyên, không lấy căn bậc hai | Obstacle recall cao hơn. Run 4 bắt được đá thật trong các mẫu minh họa nhưng sinh thêm đá giả. |
| 5–6 | Lấy căn bậc hai trọng số đá | Ít đá giả hơn và obstacle F1 nhỉnh hơn một chút, nhưng obstacle recall thấp hơn; Run 6 có mẫu tái tạo sai đá thật. |

Hiện tại chọn **không lấy căn bậc hai**: bỏ sót đá thật nguy hiểm hơn dự đoán thừa đá nếu map được dùng để lập kế hoạch sống sót. Bảng, biểu đồ W&B và ảnh reconstruction được ghi chi tiết tại [So sánh class weight của vật cản — Runs 3–6](docs/experiments/obstacle-class-weight-runs-03-06.md).

## 9. Cấu trúc mã nguồn

- `src/world_model_grid/config.py`: cấu hình môi trường, đường dẫn và siêu tham số.
- `src/world_model_grid/env.py`: luật môi trường grid.
- `src/world_model_grid/collect_data.py`: thu thập và lưu dữ liệu train/validation.
- `src/world_model_grid/model_v.py`: dataset PyTorch, Model V, train/validation, metrics và hình reconstruction.
- `src/world_model_grid/model_m.py`: phần Model M, chưa triển khai.
- `src/world_model_grid/dream_train.py`: huấn luyện trong imagined environment, chưa triển khai.
- `src/world_model_grid/evaluate.py`: đánh giá agent, chưa triển khai.
- `tests/test_env.py`: kiểm thử môi trường.
- `docs/experiments/`: ghi chú và hình ảnh thí nghiệm.
