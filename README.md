# World Model Grid

Môi trường grid 8×8 nhỏ để học và thử nghiệm world model cho bài toán học tăng cường. Mục tiêu hiện tại là giúp agent sống sót lâu nhất có thể bằng cách tránh vật cản và biên lưới.

## Nguồn cảm hứng

Ý tưởng được tham khảo từ bài báo:

> David Ha and Jürgen Schmidhuber. “World Models.” arXiv preprint arXiv:1803.10122 (2018). https://doi.org/10.48550/arXiv.1803.10122

- [Bản ghi arXiv](https://arxiv.org/abs/1803.10122)
- [Bản trình bày tương tác](https://worldmodels.github.io/)

Đây là project học tập độc lập; code hiện tại không tái tạo kiến trúc hoặc kết quả thực nghiệm của bài báo.

## Trạng thái dự án

- `GridEnv8x8` định nghĩa môi trường, state, action, reward, va chạm và giới hạn bước.
- `collect_data.py` tạo dataset train hoặc validation bằng cách trộn policy ngẫu nhiên với random walk tránh nguy hiểm ở bước kế tiếp.
- `model_v.py` có một autoencoder MLP để học mã hóa và tái tạo trạng thái dạng grid. Phần này mới học tái tạo state độc lập; chưa dự đoán chuyển trạng thái theo action.
- `model_m.py`, `dream_train.py` và `evaluate.py` hiện chưa được triển khai.

## Luật môi trường

- Lưới có kích thước 8×8, với 5–10 vật cản tĩnh và một agent ở các ô khác nhau.
- Agent có bốn hành động: lên, xuống, trái và phải.
- Mỗi bước đi an toàn nhận reward `+1`.
- Đâm vật cản hoặc đi ra ngoài lưới kết thúc episode (`terminated=True`) và nhận reward `-10`. Agent giữ nguyên vị trí ở bước va chạm.
- Nếu agent còn sống sau 100 bước, episode bị dừng do giới hạn (`truncated=True`), không phải do agent chết (`terminated=False`).

Observation là mảng `float32` có shape `(3, 8, 8)` theo thứ tự channel-first:

| Channel | Ý nghĩa |
| --- | --- |
| 0 | Ô trống |
| 1 | Vật cản |
| 2 | Agent |

## Cài đặt

Cần Python 3.14 trở lên và [uv](https://docs.astral.sh/uv/). Từ thư mục gốc repo:

```bash
uv sync
```

## Chạy test

```bash
uv run pytest
```

## Thu thập dữ liệu

Dataset không được commit vào Git: thư mục `data/` cùng các file `.npz` và `.npy` được bỏ qua trong `.gitignore`. Tạo thư mục đích nếu chưa có:

```bash
mkdir -p data
```

Tạo 100.000 episode train:

```bash
uv run python -m world_model_grid.collect_data train
```

Tạo 10.000 episode validation:

```bash
uv run python -m world_model_grid.collect_data val
```

Trong mỗi split, 70% episode chọn ngẫu nhiên giữa cả bốn hướng; 30% còn lại chọn ngẫu nhiên trong các hướng không va chạm ngay ở bước kế tiếp. Seed của train và validation được tách riêng. Dataset lần lượt được lưu tại `data/train_dataset.npz` và `data/val_dataset.npz`.

Mỗi episode có độ dài khác nhau nên các trường được lưu thành mảng NumPy `dtype=object`. Khi nạp bằng NumPy cần bật pickle:

```python
import numpy as np

with np.load("data/train_dataset.npz", allow_pickle=True) as data:
    states = data["states"]
```

Dataset giữ các trường `states`, `actions`, `rewards`, `next_states` và `terminated`. Cờ `truncated` dùng để dừng episode khi thu thập nhưng không được lưu. Collector giữ toàn bộ split trong RAM trước khi ghi file; hãy đảm bảo máy có đủ bộ nhớ và dung lượng đĩa.

## Cấu trúc mã nguồn

- `src/world_model_grid/config.py`: cấu hình môi trường, đường dẫn dữ liệu và siêu tham số Model V.
- `src/world_model_grid/env.py`: môi trường grid.
- `src/world_model_grid/collect_data.py`: tạo và lưu dataset train/validation.
- `src/world_model_grid/model_v.py`: dataset PyTorch, mô hình tái tạo state, train/validation loop và lưu checkpoint.
- `tests/test_env.py`: test luật môi trường hiện tại.
