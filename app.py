# Giới hạn thread trước khi import ML libraries.
# Đây là biện pháp deployment để tránh oversubscription trên một số máy chủ,
# KHÔNG phải yêu cầu toán học của LightGBM.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    page_icon="🛒",
    layout="wide",
)

BUNDLE_FILE = "purchase_model_bundle.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_FILE)


def sigmoid(x):
    # ổn định số học
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def restore_lgbm_categories(df, bundle):
    """
    Khôi phục ĐÚNG category dtype mà LightGBM đã học.
    Không đổi 2 -> "2", False -> "False".
    """
    df = df.copy()
    for col in bundle["categorical_features"]:
        levels = bundle["category_levels"][col]
        df[col] = pd.Categorical(df[col], categories=levels)
    return df


def make_baseline_row(bundle):
    row = dict(bundle["defaults"])
    # bảo đảm đúng thứ tự feature
    return {col: row[col] for col in bundle["feature_order"]}


def score_one(model_name, input_df, bundle):
    model = bundle["models"][model_name]

    if model_name == "LightGBM":
        input_df = restore_lgbm_categories(input_df, bundle)

    prob = float(model.predict_proba(input_df)[0, 1])
    threshold = float(bundle["metrics"][model_name]["threshold"])
    pred = int(prob >= threshold)
    return prob, threshold, pred


def local_lgbm_contributions(input_df, bundle):
    """
    LightGBM pred_contrib=True trả về SHAP-style contributions
    trên RAW SCORE (log-odds / margin) scale.
    Tổng feature contributions + expected value = raw score.
    """
    model = bundle["models"]["LightGBM"]
    x = restore_lgbm_categories(input_df, bundle)

    contrib = model.booster_.predict(x, pred_contrib=True)[0]
    feature_names = bundle["feature_order"]

    values = contrib[:-1]
    expected = float(contrib[-1])
    raw_score = expected + float(np.sum(values))

    out = pd.DataFrame({
        "feature": feature_names,
        "contribution_raw_score": values.astype(float),
        "abs_contribution": np.abs(values.astype(float)),
    }).sort_values("abs_contribution", ascending=False)

    return expected, raw_score, out


def fmt_pct(x):
    return f"{100*x:.1f}%"


try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Không thể load `{BUNDLE_FILE}`: {e}")
    st.stop()


st.title("🛒 Dự đoán khả năng mua hàng — Tree Ensembles")
st.caption(
    "Decision Tree → Bagging → Random Forest → LightGBM. "
    "Mục tiêu: hiểu được mô hình, demo được, và map được code ↔ toán."
)

tabs = st.tabs([
    "1️⃣ Bài toán & Dữ liệu",
    "2️⃣ So sánh Model",
    "3️⃣ Live Prediction",
    "4️⃣ Giải thích Prediction",
    "5️⃣ Code ↔ Toán",
    "6️⃣ Audit Notes",
])


# ==========================================================
# TAB 1
# ==========================================================
with tabs[0]:
    summary = bundle["dataset_summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số phiên truy cập", f"{summary['n_rows']:,}")
    c2.metric("Số feature", summary["n_features"])
    c3.metric("Khách mua", f"{summary['positive']:,}")
    c4.metric("Tỷ lệ mua", fmt_pct(summary["positive_rate"]))

    st.markdown("""
### Câu hỏi của bài toán

Từ hành vi của **một phiên truy cập website**, dự đoán:

\[
Y = Revenue \in \{0,1\}
\]

- \(Y=1\): phiên truy cập kết thúc bằng mua hàng.
- \(Y=0\): không mua.

Đây là **binary classification** và dữ liệu bị **mất cân bằng lớp**, nên không nên chỉ nhìn Accuracy.
""")

    st.markdown("### Cách chia dữ liệu")
    split_df = pd.DataFrame({
        "Tập": ["Train", "Validation", "Test"],
        "Số mẫu": [
            summary["split"]["train"],
            summary["split"]["validation"],
            summary["split"]["test"],
        ],
        "Dùng để làm gì": [
            "Fit tham số/model",
            "Chọn threshold + early stopping",
            "Đánh giá cuối cùng, không tune",
        ],
    })
    st.dataframe(split_df, use_container_width=True, hide_index=True)

    st.info(
        "Điểm thiết kế quan trọng: threshold được chọn trên Validation, "
        "không dùng Test để tune."
    )

    st.warning(bundle["notes"]["pagevalues_warning"])


# ==========================================================
# TAB 2
# ==========================================================
with tabs[1]:
    st.subheader("So sánh công bằng 4 họ model")

    metrics = pd.DataFrame(bundle["metrics"]).T.reset_index()
    metrics = metrics.rename(columns={"index": "Model"})

    show_cols = [
        "Model", "pr_auc", "roc_auc", "precision", "recall",
        "f1", "accuracy", "threshold"
    ]
    display = metrics[show_cols].copy()
    for col in show_cols[1:]:
        display[col] = display[col].astype(float).round(4)

    st.dataframe(
        display.sort_values("pr_auc", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"Model được đề xuất theo **PR-AUC**: "
        f"**{bundle['recommended_model']}**"
    )

    metric_to_plot = st.selectbox(
        "Metric để so sánh",
        ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"],
    )

    fig = px.bar(
        metrics.sort_values(metric_to_plot),
        x=metric_to_plot,
        y="Model",
        orientation="h",
        text=metric_to_plot,
        title=f"So sánh {metric_to_plot.upper()}",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
### Vì sao cần 4 model?

**Decision Tree**  
Một cây duy nhất, dễ hiểu nhưng variance cao.

**Bagging**  
Nhiều cây học trên các bootstrap samples độc lập rồi vote.

\[
\hat y = mode(h_1(x),...,h_B(x))
\]

**Random Forest**  
Bagging + random subset of features tại mỗi split → các cây bớt tương quan.

**Boosting / LightGBM**  
Các cây được thêm **tuần tự**, cây sau sửa phần lỗi còn lại của ensemble trước:

\[
F_m(x)=F_{m-1}(x)+\eta f_m(x)
\]
""")

    st.markdown("### LightGBM Feature Importance")

    imp = bundle["lgbm_feature_importance"].copy()
    importance_type = st.radio(
        "Kiểu importance",
        ["Gain", "Split"],
        horizontal=True,
    )

    if importance_type == "Gain":
        plot_df = imp.nlargest(12, "gain_pct").sort_values("gain_pct")
        fig = px.bar(
            plot_df,
            x="gain_pct",
            y="feature",
            orientation="h",
            text="gain_pct",
            title="Gain importance — tổng mức cải thiện objective",
        )
        fig.update_traces(texttemplate="%{text:.1f}%")
        st.caption(
            "Gain ≠ hướng tác động. Gain cao chỉ nói feature giúp giảm objective nhiều."
        )
    else:
        plot_df = imp.nlargest(12, "split").sort_values("split")
        fig = px.bar(
            plot_df,
            x="split",
            y="feature",
            orientation="h",
            text="split",
            title="Split importance — số lần feature được dùng để chia nhánh",
        )
        st.caption(
            "Split importance không có nghĩa feature làm xác suất tăng/giảm."
        )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# TAB 3
# ==========================================================
with tabs[2]:
    st.subheader("Live Prediction")

    model_name = st.selectbox(
        "Chọn model",
        list(bundle["models"].keys()),
        index=list(bundle["models"].keys()).index(bundle["recommended_model"]),
    )

    mode = st.radio(
        "Chế độ nhập",
        ["Simple Demo", "Full 17 Features"],
        horizontal=True,
    )

    baseline = make_baseline_row(bundle)

    st.caption(
        "Simple Demo: feature không nhập sẽ dùng median/mode của TRAIN. "
        "App không tự bịa giá trị như `ExitRates = BounceRates + 0.01`."
    )

    with st.form("prediction_form"):
        row = dict(baseline)

        if mode == "Simple Demo":
            c1, c2 = st.columns(2)

            with c1:
                row["ProductRelated"] = st.number_input(
                    "Số trang sản phẩm đã xem",
                    min_value=0,
                    value=int(baseline["ProductRelated"]),
                    step=1,
                )
                row["ProductRelated_Duration"] = st.number_input(
                    "Thời gian xem trang sản phẩm (giây)",
                    min_value=0.0,
                    value=float(baseline["ProductRelated_Duration"]),
                    step=10.0,
                )
                row["PageValues"] = st.number_input(
                    "PageValues",
                    min_value=0.0,
                    value=float(baseline["PageValues"]),
                    step=1.0,
                    help=(
                        "Feature rất mạnh. Cần xác nhận nó có sẵn tại đúng "
                        "thời điểm business muốn dự đoán."
                    ),
                )
                row["BounceRates"] = st.number_input(
                    "BounceRates",
                    min_value=0.0,
                    value=float(baseline["BounceRates"]),
                    step=0.001,
                    format="%.4f",
                )
                row["ExitRates"] = st.number_input(
                    "ExitRates",
                    min_value=0.0,
                    value=float(baseline["ExitRates"]),
                    step=0.001,
                    format="%.4f",
                )

            with c2:
                levels = bundle["category_levels"]
                row["Month"] = st.selectbox(
                    "Month",
                    levels["Month"],
                    index=levels["Month"].index(baseline["Month"]),
                )
                row["VisitorType"] = st.selectbox(
                    "VisitorType",
                    levels["VisitorType"],
                    index=levels["VisitorType"].index(baseline["VisitorType"]),
                )
                row["Weekend"] = st.selectbox(
                    "Weekend",
                    levels["Weekend"],
                    index=levels["Weekend"].index(baseline["Weekend"]),
                )
                row["Region"] = st.selectbox(
                    "Region",
                    levels["Region"],
                    index=levels["Region"].index(baseline["Region"]),
                )

        else:
            st.markdown("#### Numeric features")
            cols = st.columns(2)
            for i, col in enumerate(bundle["numeric_features"]):
                val = baseline[col]
                row[col] = cols[i % 2].number_input(
                    col,
                    value=float(val),
                    step=1.0 if float(val).is_integer() else 0.01,
                    key=f"full_num_{col}",
                )

            st.markdown("#### Categorical features")
            cols = st.columns(2)
            for i, col in enumerate(bundle["categorical_features"]):
                levels = bundle["category_levels"][col]
                default = baseline[col]
                idx = levels.index(default) if default in levels else 0
                row[col] = cols[i % 2].selectbox(
                    col,
                    levels,
                    index=idx,
                    key=f"full_cat_{col}",
                )

        submitted = st.form_submit_button(
            "Dự đoán",
            use_container_width=True,
        )

    if submitted:
        input_df = pd.DataFrame(
            [[row[col] for col in bundle["feature_order"]]],
            columns=bundle["feature_order"],
        )

        prob, threshold, pred = score_one(
            model_name,
            input_df,
            bundle,
        )

        st.session_state["last_input"] = input_df
        st.session_state["last_model"] = model_name
        st.session_state["last_prob"] = prob

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Model score cho lớp Mua",
            fmt_pct(prob),
        )
        c2.metric("Decision threshold", f"{threshold:.3f}")
        c3.metric(
            "Prediction",
            "MUA (1)" if pred else "KHÔNG MUA (0)",
        )

        st.progress(max(0.0, min(1.0, prob)))

        st.warning(
            "Không tự động gọi score này là 'xác suất mua thật'. "
            "Đặc biệt LightGBM dùng class weighting; muốn diễn giải xác suất "
            "cần đánh giá/calibrate probability."
        )

        with st.expander("Xem input thực sự đưa vào model"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)


# ==========================================================
# TAB 4
# ==========================================================
with tabs[3]:
    st.subheader("Giải thích prediction thật — không dùng rule viết tay")

    if "last_input" not in st.session_state:
        st.info(
            "Hãy chạy một prediction ở tab **Live Prediction** trước."
        )
    else:
        input_df = st.session_state["last_input"]

        st.markdown("""
Phần này dùng **native LightGBM contribution (`pred_contrib=True`)**.
Đây là SHAP-style decomposition trên **raw-score scale**:

\[
F(x)=E[F(x)] + \sum_j \phi_j
\]

sau đó:

\[
score = \sigma(F(x))
\]

- \(\phi_j>0\): feature đẩy model về phía lớp **Mua**.
- \(\phi_j<0\): feature đẩy model về phía **Không mua**.
""")

        expected, raw_score, contrib_df = local_lgbm_contributions(
            input_df,
            bundle,
        )
        lgb_score = sigmoid(raw_score)

        c1, c2, c3 = st.columns(3)
        c1.metric("Base raw score", f"{expected:.4f}")
        c2.metric("Final raw score", f"{raw_score:.4f}")
        c3.metric("Sigmoid(raw score)", fmt_pct(lgb_score))

        top_n = st.slider("Số feature muốn xem", 5, 17, 10)
        top = contrib_df.head(top_n).copy()
        top["direction"] = np.where(
            top["contribution_raw_score"] >= 0,
            "Đẩy về Mua",
            "Đẩy về Không mua",
        )
        top = top.sort_values("contribution_raw_score")

        fig = px.bar(
            top,
            x="contribution_raw_score",
            y="feature",
            orientation="h",
            color="direction",
            title="Local feature contributions của chính prediction này",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            contrib_df[
                ["feature", "contribution_raw_score"]
            ].round(5),
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "Khác với các câu kiểu 'tháng 11 thì cộng điểm', biểu đồ trên "
            "được lấy trực tiếp từ model cho chính input đang dự đoán."
        )


# ==========================================================
# TAB 5
# ==========================================================
with tabs[4]:
    st.subheader("Code ↔ Toán ↔ Ý nghĩa")

    with st.expander("🌳 Decision Tree", expanded=True):
        st.markdown(r"""
**Code**
```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    class_weight="balanced"
)
```

**Toán**

Với binary classification:

\[
Gini(S)=1-p_0^2-p_1^2
\]

Một split tốt làm impurity sau khi chia giảm nhiều.

**Mapping**

- `criterion="gini"` ↔ dùng Gini impurity.
- `max_depth=5` ↔ giới hạn độ phức tạp cây.
- `class_weight="balanced"` ↔ lỗi của lớp thiểu số được tăng trọng số.
""")

    with st.expander("👜 Bagging"):
        st.markdown(r"""
**Code**
```python
BaggingClassifier(
    estimator=base_tree,
    n_estimators=120,
    bootstrap=True
)
```

**Toán / thuật toán**

Tạo \(B\) bootstrap datasets:

\[
D_1,\dots,D_B \sim Bootstrap(D)
\]

train \(B\) cây:

\[
h_1,\dots,h_B
\]

và phân lớp bằng majority vote:

\[
\hat y = mode(h_1(x),...,h_B(x))
\]

**Ý nghĩa**

Các cây được train gần như độc lập → trung bình/vote giúp giảm variance.
""")

    with st.expander("🌲 Random Forest"):
        st.markdown(r"""
**Code**
```python
RandomForestClassifier(
    n_estimators=250,
    bootstrap=True,
    max_features="sqrt"
)
```

**Ý tưởng**

Random Forest = Bagging + random feature subset.

Nếu có \(p\) feature, `"sqrt"` thường cân nhắc xấp xỉ:

\[
m \approx \sqrt{p}
\]

feature tại mỗi split.

**Ý nghĩa**

Giảm tương quan giữa các cây → ensemble thường ổn định hơn Bagging cây thuần.
""")

    with st.expander("🚀 LightGBM / Gradient Boosting"):
        st.markdown(
            rf"""
**Code đang train**
```python
LGBMClassifier(
    objective="binary",
    learning_rate=0.03,
    n_estimators=500,
    max_depth=5,
    num_leaves=25,
    scale_pos_weight={bundle["scale_pos_weight"]:.4f},
    reg_lambda=1.0
)
```

### 1. Boosting update

\[
F_m(x)=F_{{m-1}}(x)+\eta f_m(x)
\]

Mapping:

- `learning_rate=0.03` ↔ \(\eta=0.03\)
- `n_estimators` ↔ số boosting rounds tối đa.
- early stopping ↔ chọn số round dựa trên Validation.

### 2. Binary log-loss

\[
L_i
=
-w_i
\left[
y_i\log p_i + (1-y_i)\log(1-p_i)
\right]
\]

\[
p_i=\sigma(F(x_i))
\]

Trong model này:

\[
w_i=
\begin{{cases}}
{bundle["scale_pos_weight"]:.4f}, & y_i=1\\
1, & y_i=0
\end{{cases}}
\]

`scale_pos_weight` là **class weight**, KHÔNG phải \(\lambda\).

### 3. Regularization

`reg_lambda=1.0` mới tương ứng với L2 regularization trên leaf weights.

### 4. Gradient + Hessian

LightGBM dùng thông tin đạo hàm bậc một \(g_i\) và bậc hai \(h_i\)
để đánh giá split / leaf update.

Một dạng lõi của split gain:

\[
Gain
\approx
\frac12
\left[
\frac{{G_L^2}}{{H_L+\lambda}}
+
\frac{{G_R^2}}{{H_R+\lambda}}
-
\frac{{G^2}}{{H+\lambda}}
\right]
\]

với:

\[
G=\sum_i g_i,\qquad H=\sum_i h_i
\]

Đây là công thức cốt lõi để hiểu; implementation thực tế còn có thêm
các constraint/hyperparameter khác.
"""
        )

    st.markdown("### Bảng mapping nhanh")
    mapping = pd.DataFrame([
        ["criterion='gini'", r"$1-\sum_k p_k^2$", "Độ không thuần của node"],
        ["bootstrap=True", r"$D_b\sim Bootstrap(D)$", "Lấy mẫu có hoàn lại"],
        ["n_estimators=B", r"$h_1,\ldots,h_B$", "Số cây / learners"],
        ["max_features='sqrt'", r"$m\approx\sqrt p$", "Random subset feature ở Forest"],
        ["learning_rate=η", r"$F_m=F_{m-1}+\eta f_m$", "Độ lớn correction của mỗi cây boosting"],
        ["scale_pos_weight", r"$w_{y=1}>w_{y=0}$", "Tăng trọng số lớp Mua"],
        ["reg_lambda", r"$\lambda$", "L2 regularization của LightGBM"],
        ["predict_proba", r"$\sigma(F(x))$", "Model output score trên [0,1]"],
    ], columns=["Code", "Toán", "Ý nghĩa"])
    st.dataframe(mapping, use_container_width=True, hide_index=True)


# ==========================================================
# TAB 6
# ==========================================================
with tabs[5]:
    st.subheader("Audit Notes — những lỗi bản cũ đã được sửa")

    st.markdown("""
1. **Không stringify categorical trước khi predict.**  
   `2` không được biến thành `"2"`, `True` không thành `"True"`.

2. **Không bịa feature ẩn.**  
   Không còn `ExitRates = BounceRates + 0.01`. Simple Demo dùng median/mode của **train** và nói rõ điều đó.

3. **Không giả lập reasoning bằng if/else.**  
   Local explanation được lấy trực tiếp từ LightGBM contribution.

4. **Không gọi `scale_pos_weight` là lambda.**  
   Class weighting và L2 regularization là hai khái niệm khác nhau.

5. **Không gọi split importance là “mức độ tác động”.**  
   App tách rõ Gain importance và Split importance.

6. **Không tune threshold trên Test.**  
   Validation chọn threshold; Test chỉ báo cáo cuối cùng.

7. **Không khẳng định raw model score là xác suất thực tế đã calibration.**

8. **Giữ cảnh báo về `PageValues`.**  
   Phải xác định feature này có tồn tại tại thời điểm business muốn dự đoán hay không.
""")

    st.markdown("### Checklist trước khi bảo vệ")
    checks = [
        "Tôi giải thích được vì sao dữ liệu mất cân bằng.",
        "Tôi giải thích được Tree → Bagging → Random Forest → Boosting.",
        "Tôi map được từng hyperparameter chính sang thuật toán/toán.",
        "Tôi phân biệt Gain importance với local contribution.",
        "Tôi không dùng Test để chọn model parameter/threshold.",
        "Tôi biết tại sao PageValues cần kiểm tra leakage / prediction timing.",
        "Tôi không gọi model score là calibrated probability khi chưa kiểm chứng.",
    ]
    for x in checks:
        st.checkbox(x, value=False)
