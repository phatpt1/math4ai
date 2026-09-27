# Online Shopper Purchase Prediction

Project được thiết kế theo 3 mục tiêu:

1. **Hiểu và demo được bài toán trên Streamlit**
2. **Thuyết trình được bằng cách mapping code ↔ toán**
3. **Tường minh: người mới đọc vẫn theo được pipeline**

## Yêu cầu

- Python **3.12**
- Dependencies được pin cứng trong `requirements.txt`

## Cấu trúc

```text
.
├── app.py                  # Streamlit entry point
├── train_models.py         # CLI: train 4 model, lưu bundle + metrics
├── requirements.txt
├── online_shoppers.csv
├── src/                    # Logic ML (không phụ thuộc UI)
│   ├── config.py           # Hằng số, danh sách feature, đường dẫn
│   ├── data.py             # Validate, chuẩn hóa, chia train/val/test
│   ├── evaluation.py       # Chọn threshold, tính metrics
│   ├── models.py           # Build + train 4 model, chọn model
│   ├── inference.py        # Predict, LightGBM contribution, what-if
│   ├── storage.py          # Serialize bundle, xuất metrics JSON
│   └── github_client.py    # Load model đã lưu, push lên GitHub
├── ui/                     # Giao diện Streamlit
│   ├── sidebar.py          # Session state, upload dataset, nút Train
│   ├── components.py       # Demo trực quan kết quả dự đoán
│   └── tabs/               # 7 tab của app
├── models/                 # purchase_model_bundle.joblib
└── artifacts/              # latest_metrics.json
```

4 model được so sánh:

- Decision Tree
- Bagging
- Random Forest
- LightGBM

## Pipeline

```text
online_shoppers.csv
        ↓
Schema + categorical typing
        ↓
Train / Validation / Test = 70 / 15 / 15 (stratified)
        ↓
Decision Tree
Bagging
Random Forest
LightGBM
        ↓
Validation: chọn threshold (max F1) + chọn model (AP → F1 → Recall)
        ↓
Test: chỉ báo cáo metrics cuối
        ↓
models/purchase_model_bundle.joblib + artifacts/latest_metrics.json
        ↓
Streamlit
```

UI và CLI dùng chung một logic train (`src/models.py → train_everything`), nên kết quả giống nhau.

## Cài đặt

```bash
python3.12 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

## Chạy

Chạy tất cả lệnh từ **thư mục gốc** của project.

```bash
python train_models.py
streamlit run app.py
```

- `train_models.py` lưu model vào `models/` và metrics vào `artifacts/`.
- `app.py` tự load model đã lưu → Predict được ngay, không cần train lại.
- Nút **Re-train** trên sidebar chỉ giữ model trong phiên hiện tại (RAM).

Tùy chỉnh đường dẫn:

```bash
SHOPPERS_CSV="online_shoppers(1).csv" python train_models.py
MODEL_BUNDLE="models/custom_bundle.joblib" python train_models.py
```

## Deploy Streamlit Community Cloud

- Main file: `app.py`
- Python version (Advanced settings): **3.12**
- Commit đủ `src/`, `ui/`, `models/`

GitHub auto-update (tùy chọn) — nút **Re-train + cập nhật GitHub** cần secrets:

```toml
# .streamlit/secrets.toml (không commit file này)
GITHUB_TOKEN = "YOUR_TOKEN"
GITHUB_REPO = "owner/repo"
GITHUB_BRANCH = "main"
```

Token chỉ cần quyền **Contents: Read and write** cho đúng repo.

## Nguyên tắc thiết kế

- Không biến categorical số/bool thành string lúc inference.
- Không hard-code dữ liệu giả như `ExitRates = BounceRates + 0.01`.
- Không dùng if/else tự viết để giả làm reasoning của LightGBM.
- Local explanation của LightGBM lấy từ `pred_contrib=True`; what-if sensitivity dùng cho cả 4 model.
- Tách Gain importance và Split importance.
- `scale_pos_weight` được giải thích đúng là class weighting.
- Validation riêng để chọn threshold, early stopping và chọn model.
- Test chỉ dùng để đánh giá cuối, không dùng để chọn model.
- So sánh Tree → Bagging → Random Forest → Boosting trên cùng một split.
- Không gọi model score là calibrated probability nếu chưa calibration.

## Điểm phải kiểm tra trước khi bảo vệ

### PageValues

`PageValues` thường rất mạnh trong dataset Online Shoppers.
Phải trả lời câu hỏi:

> Tại thời điểm business muốn dự đoán, PageValues đã có sẵn chưa?

Nếu chưa có, nên train thêm một thí nghiệm **without PageValues** và so sánh.
Đây là vấn đề prediction timing / leakage risk, không chỉ là vấn đề accuracy.

## Metric chính

Do target bị mất cân bằng, ưu tiên:

- **AP (Average Precision)**: tiêu chí chính để chọn model
- Recall
- F1
- ROC-AUC

Accuracy chỉ là metric phụ.

## Ý tưởng thuyết trình

```text
1 tree
   ↓ variance cao
Bagging
   ↓ random thêm feature
Random Forest
   ↓ thay parallel bằng sequential correction
Gradient Boosting / LightGBM
```

Đây là câu chuyện thuật toán chính của project.