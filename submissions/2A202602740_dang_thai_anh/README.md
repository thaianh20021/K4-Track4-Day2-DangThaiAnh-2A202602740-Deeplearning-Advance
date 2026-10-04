# Bài Làm Thực Hành Day 2 - Deep Learning Nâng Cao
- **Học viên:** Đặng Thái Anh
- **MSSV:** 2A202602740
- **Kho lưu trữ GitHub:** https://github.com/thaianh20021/K4-Track4-Day2-Deeplearning-Advance

## Hướng dẫn tái lập kết quả (Reproduction Guide)
```powershell
# Bước 1: So sánh 5 backbone (B01 - B05)
python run_step1.py

# Bước 2: Tinh chỉnh công thức huấn luyện (T01 - T07)
python run_step2.py

# Bước 4: Đánh giá chung cuộc trên tập Test
python run_step4.py --seeds 0

# Tự chấm điểm với eval.py
python eval.py grade --final predictions/F01_seed0_test.csv --baseline predictions/T00_seed0_test.csv --test-csv data/labels/test_subset0.csv --final-val predictions/F01_seed0_val.csv --val-csv data/labels/val_subset0.csv --labels data/labels/labels.csv
```
