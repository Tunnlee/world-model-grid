Ngày 1:
    Đã học:
    - Nhập môn Numpy
    - Cách viết một layer ẩn đơn giản trong neural network (xong)
    - Cách đọc config và thiết lập đường dẫn' (xong)
    - Học thêm Fancy Indexing
    Đã làm:
    - Setup sơ bộ Grid
    - Setup reset() và get_state()
    Câu hỏi:
    - RL thật sự là gì? Các quy ước chuẩn là như thế nào?
    - State đóng vai trò gì?


Ngày 2:
    Đã học:
    - Quy ước mypy
    - Sơ lược về MDP (Agent và action, environment và state): Tìm policy (state này thì action gì) để maximize goal (reward tích lũy)
    - Sơ lược về con trỏ và cách ghi file
    Đã làm:
    - Code step(action) và ghi test_env
    - Code collect_data và thu thập data vào dataset.npz


Ngày 3:
    Viết Docstring và README

Ngày 4:
    Đã làm:
        Code autoencoder from scratch để chuẩn bị kiến thức nền cho model V.
    Đã học:
        Cấu trúc OOP của một số class Pytorch
        Cách autograd và optimizer hoạt động

Ngày 5:
    Đã làm:
        Thu thập dữ liệu cho model_v
        Viết class kế thừa Dataset và class Dataloader
    Đã học:
        Sơ bộ về class kế thừa Dataset 
        
Ngày 6:
    Đã học:
        Khái niệm one hot
        Cách softmax và Cross Entropy Loss hoạt động
    Câu hỏi:
        Non-linear activation đóng vai trò gì? Có cách nào diễn giải cách nó hoạt động một cách trực quan không?
        Cách thuật toán optimizer Adam hoạt động
    Đã làm:
        Tạo class ModelV và huấn luyện thử

Ngày 7: 
    Đã học:
        Cách tinh chỉnh learning rate
        Cách theo dõi loss
        Cách log wandb
    Đã làm:
        Run thử vài lần với các learning rate và batch size khác nhau
        Đánh giá kết quả
        Log các thông số vào wandb

Ngày 8:
    Đã làm:
        Thu về metric của mô hình sau huấn luyện
    Nhận thấy:
        - Mô hình đang bị đoán dư đá, có thể là do class weight của đá trong crossentropyloss khá cao, hoặc là do latent space 16 chiều không đủ để nén hết thông tin của ô đá xuất hiện ngẫu nhiên cả về tần suất và vị trí.
        - Agent accuracy đã đủ tốt (0.9979)


Ngày 8:
    Đã làm:
        Phân luồng logic để chạy modelv
        Thực nghiệm so sánh khi căn bậc hai giá trị class weight
    Đã học:
        Cách sử dụng cờ -m và ảnh hưởng đến import module trong python

Ngayf 9:
    Đã làm:
        Chuyển states và next_states về dạng (16,) sau khi đưa qua encoder của model V
    Đã học:
        Cách tổ chức file và cách device hoạt động
        Batchnorm và LayerNorm 
        Sự ra đỏi của GELU và SiLU
    Câu hỏi:
        Làm thế nào hình dung cách neural network và non-linear activation hoạt động trong không gian?
        Cách vẽ hình  🎅

Ngày 10:
    Đã học:
        Affine transformation và non-linear trong mạng neuron
        Các khái niệm về không gian, manifolds
        Hình dung được cách không gian bị tác động trong mạng neuron
    Đã làm:
        Code class PredDataset và ModelM
    Hướng đi
        Học thêm về fancy indexing

Ngày 11:
    Đã học:
        Cách hoạt động của BCEWithLogitsLoss
        Cách quản lý chiều của Tensor
    Đã làm:
        Code các hàm train, val, main, comput_loss và comput_pos_weight
        Rút ngắn logic của một số hàm
    Rút ra:
        Mạng neural sử dụng affine transformation và non-inear activation để tìm ra quy luật của dữ liệu.
    Hướng đi:
        Giảm thời gian train + indexing

Ngày 12:
    Đã học:
        Cách tạo Defaultdict
        Cách quản lý dữ liệu trong class 
        Cách quản lý đồ thị tính toán bằng detach
    Đã làm:
        Viết Class MetricTracker
        Tối ưu thời gian bằng cách chuyển data lên gpu và tăng batch_size
    Hướng đi:
        Code tiếp compute() và val rollout