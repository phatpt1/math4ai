# Online Shopper Purchase Prediction

Project được thiết kế theo 3 mục tiêu:

1. **Hiểu và demo được bài toán trên Streamlit**
2. **Thuyết trình được bằng cách mapping code ↔ toán**
3. **Tường minh: người mới đọc vẫn theo được pipeline**

## Cấu trúc

- `train_models.py`: train 4 model
  - Decision Tree
  - Bagging
  - Random Forest
  - LightGBM
- `app.py`: Streamlit app
- `requirements.txt`: dependencies

## Pipeline

```text
online_shoppers.csv
        ↓
Schema + categorical typing
        ↓
Train / Validation / Test = 70 / 15 / 15
        ↓
Decision Tree
Bagging
Random Forest
LightGBM
        ↓
Validation chooses threshold
        ↓
Test reports final metrics
        ↓
Save purchase_model_bundle.joblib
        ↓
Streamlit
```

## Chạy

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Nếu file CSV có tên khác:

```bash
SHOPPERS_CSV="online_shoppers(1).csv" python train_models.py
```

## Vì sao bản này tốt hơn bản cũ?

- Không biến categorical số/bool thành string lúc inference.
- Không hard-code dữ liệu giả như `ExitRates = BounceRates + 0.01`.
- Không dùng if/else tự viết để giả làm reasoning của LightGBM.
- Local explanation lấy từ `pred_contrib=True`.
- Tách Gain importance và Split importance.
- `scale_pos_weight` được giải thích đúng là class weighting.
- Có validation riêng để chọn threshold/early stopping.
- Test chỉ dùng để đánh giá cuối.
- So sánh Tree → Bagging → Random Forest → Boosting theo cùng split.
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

- PR-AUC
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
