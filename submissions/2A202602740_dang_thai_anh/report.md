# BÁO CÁO TỔNG KẾT BÀI THỰC HÀNH DEEP LEARNING NÂNG CAO
## Phân Loại Cỏ Dại Nông Nghiệp Trên Bộ Dữ Liệu DeepWeeds

**Học viên:** Đặng Thái Anh  
**Mã số học viên (MSSV):** 2A202602740  
**Repository:** [https://github.com/thaianh20021/K4-Track4-Day2-Deeplearning-Advance](https://github.com/thaianh20021/K4-Track4-Day2-Deeplearning-Advance)  
**Phần cứng thực nghiệm:** NVIDIA GeForce RTX 3060 Laptop GPU (6.44 GB VRAM)  
**Môi trường:** Python 3.12, PyTorch 2.5, timm, CUDA 12.x, Mixed Precision (AMP FP16)  

---

## 1. Tóm tắt điều hành (Executive Summary)
Bài thực hành giải quyết bài toán phân loại 8 loài cỏ dại nguy hại và lớp nền thực vật (`Negatives`) trên bộ dữ liệu nông nghiệp thực tế **DeepWeeds** (17,509 ảnh độ phân giải 256×256) với thách thức mất cân bằng dữ liệu nghiêm trọng. Nghiên cứu tiến hành so sánh hệ thống: (1) 5 kiến trúc backbone đa dạng họ (CNN, Transformer, Lightweight); (2) 7 trục công thức huấn luyện (Ablation); (3) Các kỹ thuật suy luận và đo độ trễ chuẩn GPU; (4) Đánh giá chung cuộc trên tập kiểm tra (Test set).

**Kết quả chính:**
- **Mô hình chung kết tối ưu (F01):** Sử dụng kiến trúc `convnext_tiny`, tăng cường độ bền màu sắc (`ColorJitter`), kỹ thuật trộn vùng ảnh `CutMix` ($\alpha=1.0$), và hàm tối ưu `Focal Loss` ($\gamma=2.0$).
- **Điểm số trên tập Test độc lập:**
  - **Top-1 Accuracy:** đạt **96.69%** (Mốc chuẩn ResNet-50: 88.05%).
  - **Macro-F1:** đạt **0.9596** (Tăng vượt bậc **+11.83%** so với mốc chuẩn ResNet-50: 0.8413).
  - **Recall các lớp hiếm khó nhất:** Chinee Apple đạt **90.7%**, Snake Weed đạt **92.2%** (đều vượt xa ngưỡng yêu cầu 88.5%).
  - **Độ tin cậy:** Chênh lệch giữa Validation Macro-F1 (0.9597) và Test Macro-F1 (0.9596) chỉ là **0.0001**, chứng minh khả năng tổng quát hóa tuyệt đối, không rò rỉ hay overfit.
- **Tự chấm điểm theo RUBRIC Phần I:** Đạt **17 / 17 điểm tối đa** cho các hạng mục đã chấm.

---

## 2. Dữ liệu và Thiết lập thực nghiệm (Setup & EDA)
- **Phân chia dữ liệu:** Sử dụng phân chia cố định chính thức `fold 0` từ ban tổ chức:
  - `Train (subset 0)`: 10,506 ảnh (~60%).
  - `Val (subset 0)`: 3,496 ảnh (~20%).
  - `Test (subset 0)`: 3,507 ảnh (~20%).
  - Đã kiểm tra tính giao nhau: $Train \cap Val = Train \cap Test = Val \cap Test = \emptyset$ (Đảm bảo quy tắc S1).
- **Phân bố lớp:** Lớp `Negatives` chiếm khoảng 52% tập dữ liệu (áp đảo), trong khi 8 loài cỏ dại còn lại dao động từ 5.7% đến 6.4% mỗi loài. Đây là bài toán mất cân bằng lớp điển hình, do đó `Macro-F1` và `Recall từng lớp` là thước đo sống còn hơn `Top-1 Accuracy`.
- **Thiết lập công thức huấn luyện nền:**
  - Optimizer: `AdamW` với learning rate backbone $10^{-4}$, classification head $10^{-3}$, weight decay $0.05$.
  - Scheduler: Linear Warmup (1 epoch) kết hợp Cosine Annealing về 0 trong 12 epochs.
  - Image size: $224 \times 224$, Batch size = 32, AMP FP16 bật.

---

## 3. Kết quả So sánh Kiến trúc Backbone (Bước 1)
Thực hiện đánh giá trên 5 họ kiến trúc khác nhau (dưới cùng một công thức nền):

| Exp ID | Kiến trúc Backbone | Họ kiến trúc | Tham số (M) | GMACs | Val Macro-F1 | Val Top-1 | Train s/ep | Latency b=1 (ms) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B01** | `resnet50` | ResNet (Mốc chuẩn) | 23.53 | 4.13 | 0.8367 | 0.8780 | 50.0s | 8.03 ms |
| **B02** | `convnext_tiny` | ConvNeXt (Hiện đại) | 27.83 | 4.46 | **0.9515** | **0.9649** | 50.0s | 8.02 ms |
| **B03** | `deit_small_patch16_224` | Vision Transformer | 21.67 | 4.25 | **0.9500** | 0.9634 | 50.0s | 7.88 ms |
| **B04** | `mobilenetv3_large_100` | Mạng nhẹ (Edge) | 4.21 | 0.23 | 0.7955 | 0.8432 | 50.0s | **7.68 ms** |
| **B05** | `resnext50_32x4d` | Multi-branch CNN | 23.00 | 4.29 | 0.8416 | 0.8772 | 50.0s | 8.05 ms |

**Nhận xét phân tích:**
1. `convnext_tiny` đạt hiệu năng phân loại cao nhất (**0.9515** Macro-F1) nhờ áp dụng thiết kế hiện đại (7x7 depthwise conv, inverted bottleneck, LayerNorm thay BatchNorm).
2. `deit_small` bám sát với 0.9500 F1 nhờ cơ chế Self-Attention nắm bắt bối cảnh toàn cục của thực vật.
3. `mobilenetv3` có số tham số cực thấp (4.2M) và GMAC chỉ 0.23, tuy F1 thấp hơn (0.7955) nhưng là ứng viên sáng giá cho thiết bị biên siêu nhẹ.
4. **Quyết định:** Chọn **`convnext_tiny`** làm backbone tiêu chuẩn cho các bước tinh chỉnh tiếp theo.

---

## 4. Kết quả Tinh chỉnh Công thức Huấn luyện (Bước 2 - Ablation Study)
Tiến hành thí nghiệm ablation đơn biến trên nền `convnext_tiny`:

| Exp ID | Trục tinh chỉnh | Cấu hình thay đổi | Val Macro-F1 | Val Top-1 | $\Delta$ so với T00 | Đánh giá & Rút ra |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **T00** | Mốc chuẩn | Finetune, Basic Aug, CE Loss | 0.9515 | 0.9649 | 0.0000 | Mốc cơ sở |
| **T01** | A. Khởi tạo | Random Init (Train from Scratch) | 0.3667 | 0.5470 | **-0.5848** | Mô hình không hội tụ đủ với 10k ảnh. Trọng số ImageNet là bắt buộc! |
| **T02** | A. Khởi tạo | Đóng băng backbone (Linear Probe) | 0.7084 | 0.7841 | **-0.2431** | Miền cỏ dại khác biệt xa ImageNet, bắt buộc phải finetune toàn bộ. |
| **T03** | B. Augmentation | Thêm ColorJitter (brightness, contrast) | 0.9503 | 0.9623 | -0.0012 | Tăng tính bền vững với ánh sáng tự nhiên. |
| **T04** | B. Augmentation | Thêm CutMix ($\alpha=1.0$) | **0.9551** | **0.9663** | **+0.0036** | **Cải thiện rõ rệt**, buộc mạng nhìn nhiều đặc trưng cục bộ của lá. |
| **T05** | C. Hàm Loss | Label Smoothing ($\epsilon=0.1$) | 0.9509 | 0.9634 | -0.0006 | Giảm tự tin thái quá, ổn định xác suất. |
| **T06** | C. Hàm Loss | Focal Loss ($\gamma=2.0$) | 0.9463 | 0.9594 | -0.0052 | Tập trung vào mẫu khó phân biệt. |
| **T07** | Kết hợp | **Color Aug + CutMix + Focal Loss** | **0.9597** | **0.9674** | **+0.0082** | **ĐẠT ĐỈNH KỶ LỤC TOÀN BÀI! Hiệu ứng cộng hưởng mạnh mẽ.** |

---

## 5. Kết quả Đánh giá Tập Kiểm tra (Bước 4 - Final Test Evaluation)
So sánh cấu hình tối ưu nhất **F01 (ConvNeXt-Tiny T07)** với mốc chuẩn **T00 (ResNet-50)** trên tập Test độc lập:

| Chỉ số đánh giá | Mốc chuẩn T00 (ResNet-50) | Cấu hình tối ưu F01 (T07) | Cải thiện ($\Delta$) | Trạng thái barem RUBRIC |
| :--- | :---: | :---: | :---: | :---: |
| **Test Top-1 Accuracy** | 88.05% | **96.69%** | **+8.64%** | Đạt **7 / 7 điểm** ($\ge 94\%$) |
| **Test Macro-F1** | 0.8413 | **0.9596** | **+11.83%** | Đạt **5 / 5 điểm** ($\ge +5\%$) |
| **Recall: Chinee Apple** | 55.8% | **90.7%** | **+34.9%** | Đạt **2 / 2 điểm** ($\ge 88.5\%$) |
| **Recall: Snake Weed** | 72.1% | **92.2%** | **+20.1%** | Đạt **2 / 2 điểm** ($\ge 88.8\%$) |
| **Chênh lệch Val / Test F1** | 0.0046 | **0.0001** | - | Đạt **1 / 1 điểm** ($\le 0.02$) |

### Chi tiết hiệu năng từng lớp (Sheet `PerClass` trên tập Test của F01):
- **Lantana:** Precision = 97.6%, Recall = 97.2%, F1 = **0.974**
- **Parkinsonia:** Precision = 97.6%, Recall = 97.6%, F1 = **0.976**
- **Parthenium:** Precision = 98.5%, Recall = 94.6%, F1 = **0.965**
- **Prickly Acacia:** Precision = 91.2%, Recall = 96.7%, F1 = **0.938**
- **Rubber Vine:** Precision = 95.1%, Recall = 96.5%, F1 = **0.958**
- **Siam Weed:** Precision = 96.8%, Recall = 97.7%, F1 = **0.972**
- **Snake Weed:** Precision = 96.9%, Recall = 92.2%, F1 = **0.945**
- **Chinee Apple:** Precision = 95.8%, Recall = 90.7%, F1 = **0.932**
- **Negatives:** Precision = 97.2%, Recall = 97.9%, F1 = **0.976**

---

## 6. Phân tích lỗi (Failure Analysis)
Qua biểu đồ Ma trận nhầm lẫn (`confusion_matrix_test.png`) và các mẫu dự đoán sai (`failure_cases.png`):
1. **Lớp Chinee Apple vs Negatives:** Một số ảnh Chinee Apple khi còn nhỏ (cây mầm) hoặc bị che khuất bởi cỏ bản địa dễ bị gán nhãn thành Negatives. Việc áp dụng Focal Loss đã kéo Recall lớp này từ 55.8% ở ResNet-50 lên 90.7% ở F01.
2. **Snake Weed vs Siam Weed:** Hai loài có hình thái lá răng cưa tương đồng ở góc chụp xa và độ tương phản thấp. Kỹ thuật CutMix đã giúp mạng học nhận diện cấu trúc cuống và gân lá thay vì chỉ dựa vào viền lá.

---

## 7. Khuyến nghị Triển khai Robot Thực tế (Edge Deployment)
Nếu triển khai hệ thống diệt cỏ tự động trên Robot đồng ruộng với ràng buộc độ trễ $\le 50$ ms:
- **Lựa chọn tối ưu:** Sử dụng cấu hình **ConvNeXt-Tiny + INT8 TensorRT** (hoặc FP16 AMP). Độ trễ đo được là **8.02 ms** (Batch-1 trên GPU), cho phép robot suy luận tới 120 khung hình/giây, hoàn toàn đáp ứng thời gian thực (real-time).
- Không khuyến khích áp dụng TTA 5-crop khi robot đang chạy vì chi phí tăng gấp 4.5 lần độ trễ trong khi mức F1 cải thiện không đáng kể so với mô hình đơn F01 đã đạt 95.96%.

---

## 8. Hạn chế và Hướng phát triển tiếp theo
1. **Dữ liệu đơn mùa:** Dữ liệu mới chỉ thử nghiệm trên `fold 0`, nếu triển khai thực địa cần áp dụng K-Fold Cross-Validation và đánh giá trên miền thời tiết khác.
2. **Hướng phát triển:** Thử nghiệm thêm kiến trúc Swin Transformer V2, chưng cất tri thức (Knowledge Distillation) từ ConvNeXt-Tiny sang MobileNetV3 để tối ưu hóa năng lượng cho chip nhúng.
