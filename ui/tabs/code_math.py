import math

import streamlit as st


def render():
    st.header("6. Code ↔ Math — học từ đầu, không học thuộc công thức")

    st.markdown(
        """
Trang này trả lời một câu duy nhất: **mỗi dòng code trong project đang đại diện cho ý tưởng toán học nào?**

Hãy đọc theo flow:

**Dữ liệu → Vector x,y → Decision Tree → Bagging → Random Forest → LightGBM → Score → Threshold → Mua/Không mua**
        """
    )

    math_tabs = st.tabs(
        [
            "0️⃣ Bản đồ tổng",
            "1️⃣ Data → Vector",
            "2️⃣ Decision Tree",
            "3️⃣ Bagging",
            "4️⃣ Random Forest",
            "5️⃣ LightGBM",
            "6️⃣ Prediction & Metrics",
        ]
    )

    with math_tabs[0]:
        st.subheader("Bản đồ toàn bộ project")
        st.code(
            """Dataset (12,330 sessions)
        ↓
Preprocessing
        ↓
Split: Train 70% | Validation 15% | Test 15%
        ↓
Train 4 models
  ├─ Decision Tree
  ├─ Bagging
  ├─ Random Forest
  └─ LightGBM
        ↓
Validation: chọn threshold + chọn model theo AP
        ↓
Khóa model thắng cuộc
        ↓
Test: báo cáo cuối
        ↓
New customer → score → threshold → Mua / Không mua""",
            language="text",
        )
        st.success(
            "Điểm phải nhớ: Test KHÔNG dùng để chọn model. "
            "Validation mới dùng để chọn; Test chỉ kiểm tra cuối."
        )

    with math_tabs[1]:
        st.subheader("1. Một dòng dữ liệu biến thành vector như thế nào?")
        st.markdown(
            r"""
Một phiên truy cập có 17 feature. Ta viết ngắn gọn:

\[
x=[x_1,x_2,\ldots,x_{17}]
\]

Target:

\[
y=Revenue\in\{0,1\}
\]

- \(y=1\): khách mua.
- \(y=0\): khách không mua.

Ví dụ: `PageValues`, `BounceRates`, `Month`, `VisitorType`... là các thành phần của \(x\).
            """
        )
        st.code(
            'X = df.drop(columns=["Revenue"])\n'
            'y = df["Revenue"]\n\n'
            '# 70 / 15 / 15\n'
            'X_train, X_val, X_test, ... = split_data(df)',
            language="python",
        )
        st.info("Code đang làm đúng phép tách toán học: X = đầu vào, y = nhãn cần học.")

    with math_tabs[2]:
        st.subheader("2. Decision Tree — model đang chọn câu hỏi tốt nhất")
        st.markdown(
            r"""
Giả sử một node có 100 khách: 85 Không mua và 15 Mua. Node này còn lẫn hai lớp nên chưa "thuần".

Gini impurity:

\[
Gini(S)=1-\sum_k p_k^2
\]

Với bài toán 2 lớp:

\[
Gini(S)=1-p_0^2-p_1^2
\]

Tree thử nhiều câu hỏi kiểu `PageValues <= 12.5 ?` và chọn split làm impurity sau chia nhỏ nhất, hay tương đương làm **Gini giảm nhiều nhất**:

\[
\Delta Gini = Gini(parent)-\left(\frac{N_L}{N}Gini(L)+\frac{N_R}{N}Gini(R)\right)
\]
            """
        )
        st.code(
            'DecisionTreeClassifier(\n'
            '    criterion="gini",   # dùng Gini impurity\n'
            '    max_depth=5,        # giới hạn độ sâu\n'
            '    min_samples_leaf=20,\n'
            '    class_weight="balanced"\n'
            ')',
            language="python",
        )
        st.success(
            "Cách nhớ: Decision Tree = liên tục hỏi câu hỏi làm hai nhóm Mua/Không mua tách nhau rõ hơn."
        )

    with math_tabs[3]:
        st.subheader("3. Bagging — nhiều cây độc lập rồi bỏ phiếu")
        st.markdown(
            r"""
Một Decision Tree có thể thay đổi mạnh khi dữ liệu train thay đổi. Bagging giảm vấn đề này bằng cách tạo nhiều bộ dữ liệu bootstrap:

\[
D_b\sim Bootstrap(D),\quad b=1,2,\ldots,B
\]

Mỗi \(D_b\) train một cây \(h_b\). Với classification, prediction cuối được tổng hợp từ các cây:

\[
\hat y = mode\{h_1(x),h_2(x),\ldots,h_B(x)\}
\]

`bootstrap=True` nghĩa là lấy mẫu **có hoàn lại**. Một dòng có thể xuất hiện nhiều lần, dòng khác có thể không xuất hiện trong một bootstrap sample.
            """
        )
        st.code(
            'BaggingClassifier(\n'
            '    estimator=DecisionTreeClassifier(...),\n'
            '    n_estimators=120,   # B = 120 cây\n'
            '    bootstrap=True     # D_b ~ Bootstrap(D)\n'
            ')',
            language="python",
        )
        st.success(
            "Cách nhớ: Tree dễ dao động → train nhiều Tree trên nhiều bootstrap sample → vote → giảm variance."
        )

    with math_tabs[4]:
        st.subheader("4. Random Forest — Bagging + random feature")
        p_features = 17
        approx_features = math.sqrt(p_features)
        st.markdown(
            rf"""
Random Forest vẫn dùng nhiều cây + bootstrap giống Bagging, nhưng tại mỗi split nó chỉ cho cây nhìn **một tập con feature ngẫu nhiên**.

Nếu có \(p={p_features}\) feature và dùng `max_features="sqrt"`:

\[
m\approx\sqrt{{p}}=\sqrt{{{p_features}}}\approx {approx_features:.2f}
\]

Tức là tại một split, model thường chỉ xét khoảng **4 feature**, thay vì luôn xét cả 17.

Mục tiêu: làm các cây **bớt giống nhau**, tức giảm correlation giữa các cây.
            """
        )
        st.code(
            'RandomForestClassifier(\n'
            '    n_estimators=250,\n'
            '    bootstrap=True,       # giống Bagging\n'
            '    max_features="sqrt", # random feature subset\n'
            '    max_depth=8\n'
            ')',
            language="python",
        )
        st.success("Cách nhớ: Random Forest = Bagging + random feature selection.")

    with math_tabs[5]:
        st.subheader("5. LightGBM — các cây học tuần tự để cải thiện lỗi")
        st.markdown(
            r"""
Khác Bagging/Random Forest, Boosting không xây các cây hoàn toàn độc lập. Nó xây tuần tự:

\[
F_m(x)=F_{m-1}(x)+\eta f_m(x)
\]

Trong đó:

- \(F_{m-1}(x)\): ensemble trước khi thêm cây mới.
- \(f_m(x)\): cây mới.
- \(\eta\): learning rate, điều khiển cây mới đóng góp mạnh đến đâu.

Với binary classification, một dạng loss phổ biến là binary log-loss:

\[
L_i=-w_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
\]

LightGBM dùng thông tin đạo hàm của loss:

\[
g_i=\frac{\partial L}{\partial F(x_i)},\qquad
h_i=\frac{\partial^2 L}{\partial F(x_i)^2}
\]

Gradient cho biết hướng loss thay đổi; Hessian cho biết độ cong. LightGBM dùng chúng để tìm các split/cây giúp objective tốt hơn.
            """
        )
        st.code(
            'LGBMClassifier(\n'
            '    n_estimators=500,       # tối đa M vòng boosting\n'
            '    learning_rate=0.03,     # eta = 0.03\n'
            '    max_depth=5,\n'
            '    num_leaves=25,\n'
            '    reg_lambda=1.0,         # lambda: L2 regularization\n'
            '    scale_pos_weight=...    # trọng số lớp Mua\n'
            ')',
            language="python",
        )
        if st.session_state["bundle"] is not None:
            b = st.session_state["bundle"]
            st.metric(
                "scale_pos_weight hiện tại",
                f"{b['scale_pos_weight']:.3f}",
                help="N_negative / N_positive trên TRAIN",
            )
        st.warning(
            "Đừng nhầm: scale_pos_weight là class weighting; "
            "reg_lambda mới liên hệ với λ của L2 regularization."
        )
        st.success(
            "Cách nhớ: LightGBM = cây sau được thêm vào để cải thiện ensemble hiện tại; "
            "learning_rate quyết định bước cải thiện lớn hay nhỏ."
        )

    with math_tabs[6]:
        st.subheader("6. Từ score đến quyết định Mua / Không mua")
        st.markdown(
            r"""
Sau khi model tạo score \(s(x)\), ta so với threshold \(t\):

\[
\hat y=
\begin{cases}
1 & s(x)\ge t\\
0 & s(x)<t
\end{cases}
\]

Trong project, threshold được chọn trên **Validation** bằng F1 tốt nhất, sau đó khóa lại và mang sang Test.

### Ba metric chính

**Recall** — trong khách thực sự Mua, bắt được bao nhiêu:

\[
Recall=\frac{TP}{TP+FN}
\]

**F1** — cân bằng Precision và Recall:

\[
F1=2\frac{Precision\cdot Recall}{Precision+Recall}
\]

**AP (Average Precision)** — tổng hợp Precision theo các mức Recall; đây là tiêu chí chính để chọn model vì lớp Mua là lớp thiểu số.
            """
        )
        st.code(
            '# threshold chọn trên VALIDATION\n'
            'threshold = choose_threshold(y_val, val_prob)\n\n'
            '# model winner chọn bằng VALIDATION AP\n'
            'recommended_model = max(\n'
            '    validation_results,\n'
            '    key=lambda name: (\n'
            '        validation_results[name]["ap"],\n'
            '        validation_results[name]["f1"],\n'
            '        validation_results[name]["recall"],\n'
            '    ),\n'
            ')\n\n'
            '# TEST chỉ dùng báo cáo sau khi đã khóa model + threshold',
            language="python",
        )
        st.error(
            "Sai về phương pháp: dùng Test để chọn model hoặc tune threshold. "
            "Đúng: Validation chọn; Test chỉ báo cáo cuối."
        )

    with st.expander("📘 Từ điển 12 từ phải nhớ"):
        st.markdown(
            """
- **Feature**: biến đầu vào.
- **Target/Label**: nhãn cần dự đoán.
- **Train**: dữ liệu để model học.
- **Validation**: dữ liệu để chọn model/threshold/hyperparameter.
- **Test**: dữ liệu kiểm tra cuối.
- **Split**: phép chia node của cây.
- **Leaf**: node cuối của cây.
- **Bootstrap**: lấy mẫu ngẫu nhiên có hoàn lại.
- **Ensemble**: tổ hợp nhiều model.
- **Boosting**: học tuần tự để cải thiện lỗi.
- **Threshold**: ngưỡng biến score thành 0/1.
- **AP**: chất lượng Precision–Recall qua nhiều threshold.
            """
        )