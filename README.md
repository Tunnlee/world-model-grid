# World Model Grid

Môi trường grid 8×8 nhỏ để xây dựng và thử nghiệm world model cho bài toán học tăng cường. Mục tiêu hiện tại là để agent sống sót càng lâu càng tốt bằng cách tránh vật cản và biên lưới.

## Nguồn cảm hứng

Ý tưởng xây dựng một môi trường nhỏ để nghiên cứu world model được tham khảo từ bài báo:

> David Ha and Jürgen Schmidhuber. “World Models.” arXiv preprint arXiv:1803.10122 (2018). https://doi.org/10.48550/arXiv.1803.10122

- Bản ghi arXiv: https://arxiv.org/abs/1803.10122
- Bản trình bày tương tác của tác giả: https://worldmodels.github.io/

Repo này là project học tập độc lập với môi trường grid 8×8; phần triển khai hiện tại không phải bản tái tạo kiến trúc hay kết quả thực nghiệm của bài báo.

## Trạng thái hiện tại

Repo hiện có:

- Môi trường GridEnv8x8 với reset, quan sát, hành động, reward, kết thúc do va chạm và giới hạn bước.
- Script thu thập dữ liệu từ policy ngẫu nhiên và policy chọn ngẫu nhiên trong các nước đi an toàn tức thời.
- Test cho reset, state, chuyển động, va chạm, biên và giới hạn episode.

Các module model_m.py, model_v.py, dream_train.py và evaluate.py hiện là placeholder rỗng; mô hình và quy trình huấn luyện chưa được triển khai.

## Luật môi trường

- Bàn cờ có kích thước 8×8.
- Mỗi episode đặt ngẫu nhiên 5–10 vật cản và một agent vào các ô khác nhau.
- Agent có bốn hành động: lên, xuống, trái, phải.
- Bước đi an toàn nhận reward +1.
- Đâm vật cản hoặc đi ra ngoài lưới kết thúc episode (terminated=True) và nhận reward -10. Agent giữ nguyên vị trí ở bước va chạm.
- Episode còn sống sau 100 bước bị dừng do giới hạn (truncated=True), không phải do agent chết (terminated=False).

Quan sát là mảng float32 có shape (3, 8, 8), theo thứ tự channel-first:

| Channel | Ý nghĩa |
| --- | --- |
| 0 | Ô trống |
| 1 | Vật cản |
| 2 | Agent |

Mỗi ô thuộc đúng một lớp trong ba channel trên.

## Cài đặt

Cần Python 3.14 trở lên và uv (https://docs.astral.sh/uv/).

Từ thư mục gốc repo:

```bash
uv sync
```

## Chạy test

```bash
uv run pytest
```

## Thu thập dữ liệu

Dataset không được commit vào Git (thư mục data/, các file .npz và .npy đã được bỏ qua trong .gitignore). Tạo thư mục đích rồi chạy collector:

```bash
mkdir -p data
uv run python -m world_model_grid.collect_data
```

Script tạo 50.000 episode: 70% policy ngẫu nhiên trên cả bốn hướng và 30% policy chọn ngẫu nhiên trong các hướng an toàn ở bước kế tiếp. Dataset được ghi vào data/dataset.npz. Kích thước có thể lớn; hãy đảm bảo còn đủ dung lượng đĩa và RAM.

Các episode có độ dài khác nhau nên những trường được lưu dưới dạng mảng NumPy dtype=object. Khi nạp bằng NumPy, cần bật pickle, ví dụ:

```python
import numpy as np

with np.load("data/dataset.npz", allow_pickle=True) as data:
    states = data["states"]
```

Chỉ nhãn terminated hiện được lưu trong dataset. truncated được dùng để dừng episode trong lúc thu thập nhưng không được lưu; do đó dataset chưa giữ lại thông tin episode nào dừng vì chạm giới hạn 100 bước.

## Cấu trúc mã nguồn

- src/world_model_grid/config.py: kích thước, hành động, reward, đường dẫn dữ liệu và số episode.
- src/world_model_grid/env.py: định nghĩa môi trường grid.
- src/world_model_grid/collect_data.py: thu thập và lưu các transition theo episode.
- tests/test_env.py: kiểm thử các luật hiện có.
