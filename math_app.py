import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CS115 - Math for Computer Science Review",
    page_icon="📘",
    layout="wide",
)

st.title("📘 CS115 - Math for Computer Science")
st.caption("Ứng dụng ôn tập tổng hợp từ 11 tài liệu PDF đã cung cấp")

# -----------------------------
# Helpers
# -----------------------------
def box(title, content, icon="✅"):
    st.markdown(f"### {icon} {title}")
    st.markdown(content)

def formula(label, latex):
    st.markdown(f"**{label}**")
    st.latex(latex)

# -----------------------------
# Sidebar
# -----------------------------
topics = [
    "Tổng quan môn học",
    "Machine Learning",
    "Đại số tuyến tính",
    "Xác suất thống kê",
    "Giải tích vector",
    "Hình học giải tích",
    "Linear Regression",
    "Optimization",
    "Neural Network",
    "Backpropagation",
    "Tóm tắt công thức",
    "Quiz trắc nghiệm",
    "Bài tập có đáp án",
    "Visualization: Linear Regression",
    "Visualization: Gradient Descent",
]

with st.sidebar:
    st.header("Điều hướng")
    topic = st.radio("Chọn nội dung", topics)
    st.divider()
    st.markdown("**11 tài liệu được tổng hợp:**")
    st.markdown("""
- CS115_Course_Introduction
- CS115.01_Introduction to Machine Learning
- Linear Algebra Review
- Chương 1 Review ĐSTT
- C1 Review XSTK
- C1 Review Giải tích vector
- Vector Calculus
- Analytic Geometry
- Linear Regression Tutorial
- Optimization
- Neural Network
""")
    st.info("Mục tiêu: ôn nhanh lý thuyết, công thức và mối liên hệ giữa Toán và Machine Learning.")

# -----------------------------
# Topic content
# -----------------------------

if topic == "Tổng quan môn học":
    st.header("1. Tổng quan môn học CS115")
    box("Mục tiêu", """
Sau khi học xong, cần:
- Hiểu các nền tảng toán học dùng trong một số thuật toán Machine Learning cơ bản.
- Vận dụng công cụ toán để giải các bài toán ML.
- So sánh phương pháp toán và thuật toán tin học trong khoa học máy tính.
""")
    box("Các khối kiến thức chính", """
1. Đại số tuyến tính  
2. Vi tích phân / Giải tích vector  
3. Xác suất thống kê  
4. Hình học giải tích  
5. Machine Learning cơ bản  
6. Tối ưu hóa  
7. Neural Network và Backpropagation
""")
    st.success("Chuỗi tư duy quan trọng: Vector/Matrix → Gradient → Loss → Optimization → Machine Learning → Neural Network.")

elif topic == "Machine Learning":
    st.header("2. Tổng quan Machine Learning")
    box("Machine Learning là gì?", """
Machine Learning nghiên cứu các thuật toán cho phép máy tính cải thiện hiệu năng giải quyết vấn đề thông qua dữ liệu.

Theo cách mô tả kinh điển:
- **T (Task):** tác vụ cần thực hiện.
- **P (Performance):** độ đo hiệu năng.
- **E (Experience):** kinh nghiệm / dữ liệu.
""")
    box("Ba nhóm chính", """
- **Supervised Learning:** có input và output tương ứng, học ánh xạ từ input → output.
- **Unsupervised Learning:** chỉ có input, tìm cấu trúc/đặc trưng của dữ liệu.
- **Reinforcement Learning:** agent học chiến lược hành động để tối ưu kết quả trong môi trường.
""")
    box("Classification", """
Bài toán phân lớp học ánh xạ:
""")
    formula("Mô hình tổng quát", r"f_\theta: \mathcal{X} \rightarrow \mathcal{Y}")
    st.markdown("""
Ví dụ phân lớp ảnh:
- Input có thể là ảnh RGB kích thước lớn.
- Output là nhãn lớp.
- Thách thức chính: dữ liệu có số chiều rất cao.
""")
    box("Linear / Polynomial / Logistic Model", """
- Linear regression dùng hàm tuyến tính theo tham số.
- Polynomial regression tăng khả năng biểu diễn bằng feature phi tuyến.
- Logistic regression biến score thành xác suất phân lớp.
""")
    formula("Linear model", r"f(\mathbf{x};\theta)=b+\mathbf{w}^T\mathbf{x}")
    box("Overfitting", """
Mô hình quá phức tạp có thể fit dữ liệu train rất tốt nhưng dự đoán kém trên test data.

Điểm cần nhớ:
- Train error thấp chưa đủ.
- Cần đánh giá khả năng tổng quát hóa.
""")
    box("No Free Lunch", """
Không có một mô hình tốt nhất cho mọi bài toán.  
Lựa chọn mô hình phụ thuộc dữ liệu, bài toán và kết quả thực nghiệm.
""")

elif topic == "Đại số tuyến tính":
    st.header("3. Đại số tuyến tính")
    box("Vector và Matrix", """
Vector và ma trận là nền tảng biểu diễn dữ liệu và tham số trong Machine Learning.
""")
    formula("Hệ phương trình tuyến tính", r"A\mathbf{x}=\mathbf{b}")
    box("Phép toán ma trận", """
- Cộng ma trận: cùng kích thước.
- Nhân ma trận: số cột của ma trận trái = số hàng của ma trận phải.
- Ma trận đơn vị: \(AI=IA=A\).
- Chuyển vị: \(A^T\).
- Ma trận nghịch đảo: \(AA^{-1}=I\), nếu tồn tại.
""")
    formula("Phần tử tích ma trận", r"c_{ij}=\sum_k a_{ik}b_{kj}")
    box("Độc lập tuyến tính", """
Tập \(\{v_1,\dots,v_p\}\) độc lập tuyến tính nếu:
""")
    formula("Điều kiện", r"c_1v_1+\cdots+c_pv_p=0 \Rightarrow c_1=\cdots=c_p=0")
    box("Basis và Rank", """
- **Basis:** tập vector độc lập tuyến tính sinh ra toàn bộ không gian.
- **Rank:** số chiều của không gian cột / số vector độc lập tuyến tính cực đại.
""")
    box("Linear Mapping", """
Ánh xạ tuyến tính bảo toàn cộng vector và nhân vô hướng.
""")
    formula("Dạng ma trận", r"T(\mathbf{x})=A\mathbf{x}")
    box("Positive Definite Matrix", """
Một ma trận xác định dương thường được nhận biết qua:
- \(x^TAx>0\) với mọi \(x\neq 0\).
- Các trị riêng dương.
- Xuất hiện nhiều trong bài toán tối ưu hóa bậc hai và Hessian.
""")

elif topic == "Xác suất thống kê":
    st.header("4. Xác suất thống kê")
    box("Biến ngẫu nhiên", """
Biến ngẫu nhiên ánh xạ mỗi kết quả của phép thử sang một giá trị số.

Hai loại:
- Rời rạc
- Liên tục
""")
    box("Bernoulli", """
Một phép thử chỉ có hai kết quả: thành công / thất bại.
""")
    formula("Xác suất", r"P(X=1)=p,\qquad P(X=0)=1-p")
    box("Phân phối nhị thức", """
Nếu thực hiện \(n\) phép thử Bernoulli độc lập và \(X\) là số lần thành công:
""")
    formula("PMF", r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}")
    formula("Kỳ vọng", r"E[X]=np")
    formula("Phương sai", r"\mathrm{Var}(X)=np(1-p)")
    box("Vai trò trong ML", """
Xác suất dùng để:
- Mô hình hóa bất định.
- Diễn giải output như xác suất.
- Xây dựng likelihood, loss và mô hình phân lớp xác suất.
""")

elif topic == "Giải tích vector":
    st.header("5. Giải tích vector")
    box("Đạo hàm", """
Đạo hàm mô tả tốc độ biến thiên của hàm.
""")
    formula("Định nghĩa", r"f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}")
    box("Đạo hàm riêng", """
Với hàm nhiều biến, giữ các biến khác cố định và lấy đạo hàm theo một biến.
""")
    formula("Partial derivative", r"\frac{\partial f}{\partial x_i}")
    box("Gradient", """
Gradient gom toàn bộ đạo hàm riêng:
""")
    formula("Gradient", r"\nabla f(\mathbf{x})=\begin{bmatrix}\frac{\partial f}{\partial x_1}\\ \vdots\\ \frac{\partial f}{\partial x_n}\end{bmatrix}")
    st.markdown("""
Gradient chỉ hướng tăng nhanh nhất của hàm.  
\(-\nabla f\) là hướng giảm nhanh nhất cục bộ.
""")
    box("Taylor Series", """
Taylor xấp xỉ hàm quanh một điểm bằng đạo hàm của hàm tại điểm đó.
""")
    formula("Taylor bậc 2", r"f(x)\approx f(x_0)+f'(x_0)(x-x_0)+\frac12f''(x_0)(x-x_0)^2")
    box("Chain Rule", """
Chain rule là nền tảng của backpropagation.
""")
    formula("Một biến", r"\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}")
    box("Computational Graph", """
Mạng neural được mô tả bằng graph tính toán:
- Forward: tính output/loss.
- Backward: lan truyền đạo hàm ngược bằng chain rule.
""")

elif topic == "Hình học giải tích":
    st.header("6. Hình học giải tích")
    box("Norm", """
Norm đo độ dài vector.
""")
    formula("Euclidean norm", r"\|\mathbf{x}\|_2=\sqrt{\mathbf{x}^T\mathbf{x}}")
    box("Khoảng cách", "")
    formula("Euclidean distance", r"d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2")
    box("Inner Product", "")
    formula("Dot product", r"\langle \mathbf{x},\mathbf{y}\rangle=\mathbf{x}^T\mathbf{y}")
    box("Góc giữa hai vector", "")
    formula("Cosine", r"\cos\theta=\frac{\mathbf{x}^T\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}")
    box("Orthogonality", "")
    formula("Trực giao", r"\mathbf{x}^T\mathbf{y}=0")
    box("Orthonormal Basis", """
Một basis trực chuẩn có:
- Các vector đôi một trực giao.
- Mỗi vector có norm bằng 1.
""")
    box("Projection", """
Chiếu vector lên một hướng / không gian con là khái niệm quan trọng trong least squares.
""")
    formula("Projection lên vector u", r"\mathrm{proj}_{u}(v)=\frac{u^Tv}{u^Tu}u")

elif topic == "Linear Regression":
    st.header("7. Linear Regression")
    box("Mô hình", """
Hồi quy tuyến tính dự đoán output bằng tổ hợp tuyến tính của input.
""")
    formula("1 biến", r"\hat y=b+wx")
    formula("Nhiều biến", r"\hat y=b+\mathbf{w}^T\mathbf{x}")
    box("Loss / Cost", """
Mục tiêu là làm nhỏ sai số bình phương.
""")
    formula("Squared Error", r"J(\mathbf{w})=\frac12\sum_i(y^{(i)}-\hat y^{(i)})^2")
    box("Gradient Descent", "")
    formula("Update", r"\mathbf{w}_{t+1}=\mathbf{w}_t-\eta\nabla J(\mathbf{w}_t)")
    st.markdown("""
Trong đó:
- \(\eta\): learning rate.
- Gradient cho biết hướng tăng.
- Trừ gradient để giảm cost.
""")
    box("Least Squares", """
Tối ưu Linear Regression là một bài toán least-squares.  
Ý tưởng hình học liên quan trực tiếp đến projection.
""")
    box("Workflow thực hành", """
1. Chuẩn bị dữ liệu  
2. Trực quan dữ liệu  
3. Khởi tạo tham số  
4. Tính dự đoán  
5. Tính loss  
6. Tính gradient  
7. Update tham số  
8. Lặp nhiều epoch  
9. Kiểm tra cost giảm
""")

elif topic == "Optimization":
    st.header("8. Optimization")
    box("Mục tiêu", """
Tối ưu hóa tìm điều kiện làm một hàm đạt giá trị nhỏ nhất hoặc lớn nhất.
""")
    formula("Bài toán ML phổ biến", r"\min_{\theta}L(\theta)")
    box("Gradient Descent", "")
    formula("Update cơ bản", r"\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)")
    box("Ba cách dùng dữ liệu", """
- **Batch GD:** dùng toàn bộ dữ liệu mỗi update.
- **SGD:** dùng 1 mẫu mỗi update.
- **Mini-batch GD:** dùng một nhóm mẫu.
""")
    box("Các optimizer mở rộng", """
- Momentum
- Nesterov Accelerated Gradient
- Adagrad
- RMSprop
- Adam
""")
    box("Newton's Method", """
Sử dụng thông tin bậc hai (Hessian) để điều chỉnh bước cập nhật.
""")
    formula("Newton", r"\theta_{t+1}=\theta_t-H^{-1}\nabla L(\theta_t)")
    box("Convex Optimization", """
Một số lớp bài toán có thể giải hiệu quả và đáng tin cậy hơn, ví dụ:
- Linear programming
- Least squares
- Convex optimization
""")
    box("Regularization", """
Regularization hạn chế độ phức tạp mô hình.
""")
    formula("L2", r"L_{\text{reg}}=L+\lambda\|\mathbf{w}\|_2^2")
    formula("L1", r"L_{\text{reg}}=L+\lambda\|\mathbf{w}\|_1")

elif topic == "Neural Network":
    st.header("9. Neural Network")
    box("Từ Linear Classifier đến Neural Network", """
Linear classifier:
""")
    formula("Linear score", r"f(\mathbf{x},W)=W\mathbf{x}+\mathbf{b}")
    st.markdown("Neural Network thêm hidden layer và activation để học quan hệ phi tuyến.")
    formula("2-layer network", r"\mathbf{h}=\phi(W_1\mathbf{x}+b_1),\qquad \mathbf{s}=W_2\mathbf{h}+b_2")
    box("Kiến trúc mạng", """
- Single-layer feedforward
- Multi-layer feedforward
- Recurrent network
""")
    box("Hidden Layer", """
Hidden layer giúp mô hình tự học biểu diễn/đặc trưng thay vì phải thiết kế đặc trưng hoàn toàn thủ công.
""")
    box("Deep Neural Network", """
Mạng sâu có nhiều lớp.  
Các biến thể tiêu biểu trong tài liệu:
- CNN cho hình ảnh.
- RNN cho chuỗi tuần tự.
""")
    box("Huấn luyện", """
Quá trình huấn luyện:
1. Forward propagation
2. Tính loss
3. Backpropagation
4. Optimizer cập nhật weight
5. Lặp qua nhiều epoch
""")

elif topic == "Backpropagation":
    st.header("10. Backpropagation")
    box("Ý tưởng", """
Backpropagation là việc áp dụng chain rule một cách đệ quy trên computational graph để tính gradient của loss theo toàn bộ tham số.
""")
    box("Forward Pass", """
- Tính output từng node.
- Lưu các giá trị trung gian cần thiết.
""")
    box("Backward Pass", """
- Bắt đầu từ loss.
- Tính gradient ngược qua từng phép toán.
- Tính đạo hàm của loss theo input, weight, bias.
""")
    formula("Chain rule tổng quát", r"\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}")
    box("Vai trò", """
Backprop không tự update weight.  
Nó chỉ tính gradient. Optimizer mới dùng gradient để cập nhật tham số.
""")
    box("Gradient checking", """
Tài liệu nhấn mạnh:
- Numerical gradient: dễ viết nhưng chậm và xấp xỉ.
- Analytic gradient: nhanh và chính xác nhưng dễ code sai.
- Thực tế nên kiểm tra analytic gradient bằng numerical gradient.
""")

elif topic == "Tóm tắt công thức":
    st.header("11. Cheat Sheet Công thức")
    st.markdown("### Đại số tuyến tính")
    formula("Linear system", r"A\mathbf{x}=\mathbf{b}")
    formula("Dot product", r"\mathbf{x}^T\mathbf{y}=\sum_i x_i y_i")
    formula("Norm", r"\|\mathbf{x}\|_2=\sqrt{\mathbf{x}^T\mathbf{x}}")
    formula("Projection", r"\mathrm{proj}_u(v)=\frac{u^Tv}{u^Tu}u")

    st.markdown("### Xác suất")
    formula("Binomial", r"P(X=k)=\binom nk p^k(1-p)^{n-k}")
    formula("Mean", r"E[X]=np")
    formula("Variance", r"\mathrm{Var}(X)=np(1-p)")

    st.markdown("### Giải tích")
    formula("Derivative", r"f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}")
    formula("Gradient", r"\nabla f=[\partial f/\partial x_1,\dots,\partial f/\partial x_n]^T")
    formula("Chain rule", r"\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}")

    st.markdown("### Linear Regression / Optimization")
    formula("Linear regression", r"\hat y=b+\mathbf{w}^T\mathbf{x}")
    formula("Squared error", r"J=\frac12\sum_i(y_i-\hat y_i)^2")
    formula("Gradient descent", r"\theta\leftarrow\theta-\eta\nabla L(\theta)")
    formula("Newton", r"\theta\leftarrow\theta-H^{-1}\nabla L(\theta)")
    formula("L2", r"L_{\text{reg}}=L+\lambda\|\mathbf{w}\|_2^2")

    st.markdown("### Neural Network")
    formula("Layer", r"\mathbf{z}=W\mathbf{x}+\mathbf{b}")
    formula("Activation", r"\mathbf{h}=\phi(\mathbf{z})")
    formula("2-layer", r"\mathbf{s}=W_2\phi(W_1\mathbf{x}+b_1)+b_2")


elif topic == "Quiz trắc nghiệm":
    st.header("🧠 Quiz trắc nghiệm CS115")
    st.caption("20 câu hỏi tổng hợp: Đại số tuyến tính, xác suất, giải tích, ML, tối ưu và Neural Network.")

    quiz = [
        {
            "q": "1. Điều kiện để hai vector x và y trực giao là gì?",
            "options": ["x + y = 0", "xᵀy = 0", "||x|| = ||y||", "x = y"],
            "answer": 1,
            "explain": "Hai vector trực giao khi inner product của chúng bằng 0."
        },
        {
            "q": "2. Gradient của hàm nhiều biến biểu diễn điều gì?",
            "options": [
                "Hướng giảm nhanh nhất",
                "Hướng tăng nhanh nhất của hàm tại điểm đang xét",
                "Giá trị nhỏ nhất của hàm",
                "Ma trận nghịch đảo"
            ],
            "answer": 1,
            "explain": "Gradient chỉ hướng tăng nhanh nhất; negative gradient chỉ hướng giảm nhanh nhất cục bộ."
        },
        {
            "q": "3. Công thức cập nhật Gradient Descent là?",
            "options": [
                "θ ← θ + η∇L",
                "θ ← θ − η∇L",
                "θ ← η/∇L",
                "θ ← ∇L − θ"
            ],
            "answer": 1,
            "explain": "Ta đi ngược hướng gradient để giảm loss."
        },
        {
            "q": "4. Trong Linear Regression nhiều biến, mô hình có dạng nào?",
            "options": [
                "ŷ = b + wᵀx",
                "ŷ = wx²",
                "ŷ = softmax(x)",
                "ŷ = ||x||"
            ],
            "answer": 0,
            "explain": "Mô hình tuyến tính đa biến có dạng b + wᵀx."
        },
        {
            "q": "5. Overfitting là hiện tượng nào?",
            "options": [
                "Train error cao, test error thấp",
                "Train và test đều tốt",
                "Train rất tốt nhưng test kém",
                "Mô hình không có tham số"
            ],
            "answer": 2,
            "explain": "Overfitting xảy ra khi mô hình học quá sát dữ liệu train và tổng quát hóa kém."
        },
        {
            "q": "6. Phân phối Binomial mô tả gì?",
            "options": [
                "Số lần thành công trong n phép thử Bernoulli độc lập",
                "Khoảng cách giữa hai vector",
                "Đạo hàm của loss",
                "Số lớp của neural network"
            ],
            "answer": 0,
            "explain": "X ~ B(n,p) đếm số lần biến cố thành công xảy ra trong n phép thử."
        },
        {
            "q": "7. Kỳ vọng của X ~ B(n,p) là?",
            "options": ["p", "np", "n/p", "np(1-p)"],
            "answer": 1,
            "explain": "E[X] = np."
        },
        {
            "q": "8. Phương sai của X ~ B(n,p) là?",
            "options": ["np", "n²p", "np(1-p)", "p(1-p)"],
            "answer": 2,
            "explain": "Var(X) = np(1-p)."
        },
        {
            "q": "9. Backpropagation chủ yếu dựa trên quy tắc nào?",
            "options": ["Bayes rule", "Chain rule", "Cramer rule", "Cosine rule"],
            "answer": 1,
            "explain": "Backprop là ứng dụng đệ quy của chain rule trên computational graph."
        },
        {
            "q": "10. Vai trò chính của backpropagation là gì?",
            "options": [
                "Tự động cập nhật weight",
                "Tính gradient của loss theo các tham số",
                "Chia train/test",
                "Chuẩn hóa dữ liệu"
            ],
            "answer": 1,
            "explain": "Backprop tính gradient; optimizer mới dùng gradient để cập nhật weight."
        },
        {
            "q": "11. Batch Gradient Descent dùng bao nhiêu dữ liệu cho mỗi lần tính gradient?",
            "options": ["1 mẫu", "Một mini-batch", "Toàn bộ tập train", "Không dùng dữ liệu"],
            "answer": 2,
            "explain": "Batch GD tính gradient dựa trên toàn bộ tập huấn luyện."
        },
        {
            "q": "12. SGD theo định nghĩa cơ bản dùng?",
            "options": ["Một mẫu mỗi update", "Toàn bộ dữ liệu", "Chỉ validation set", "Không tính gradient"],
            "answer": 0,
            "explain": "Stochastic Gradient Descent cập nhật tham số dựa trên từng mẫu."
        },
        {
            "q": "13. Projection của v lên u có hệ số nào?",
            "options": [
                "(uᵀv)/(uᵀu)",
                "(uᵀu)/(uᵀv)",
                "||u+v||",
                "det(u)"
            ],
            "answer": 0,
            "explain": "proj_u(v) = ((uᵀv)/(uᵀu))u."
        },
        {
            "q": "14. Tập vector độc lập tuyến tính khi nào?",
            "options": [
                "Có ít nhất một vector 0",
                "Tổ hợp tuyến tính bằng 0 chỉ có nghiệm hệ số bằng 0",
                "Tất cả vector cùng phương",
                "Mọi vector có norm bằng 1"
            ],
            "answer": 1,
            "explain": "Đó là định nghĩa cơ bản của độc lập tuyến tính."
        },
        {
            "q": "15. L2 regularization thêm đại lượng nào vào loss?",
            "options": ["λ||w||²", "λ||w||₁", "λ/w", "λ det(W)"],
            "answer": 0,
            "explain": "L2 sử dụng bình phương norm Euclidean của vector trọng số."
        },
        {
            "q": "16. Neural Network khác Linear Classifier cơ bản ở điểm chính nào?",
            "options": [
                "Không có tham số",
                "Có hidden layer và phép biến đổi phi tuyến",
                "Không cần dữ liệu",
                "Không dùng gradient"
            ],
            "answer": 1,
            "explain": "Hidden layer + activation giúp mạng biểu diễn quan hệ phi tuyến."
        },
        {
            "q": "17. Newton's method sử dụng thêm thông tin nào so với Gradient Descent?",
            "options": ["Hessian / đạo hàm bậc hai", "Chỉ xác suất", "Chỉ norm L1", "Chỉ bias"],
            "answer": 0,
            "explain": "Newton sử dụng Hessian để khai thác curvature của hàm."
        },
        {
            "q": "18. Một orthonormal basis phải thỏa?",
            "options": [
                "Các vector cùng phương",
                "Các vector trực giao đôi một và norm bằng 1",
                "Tổng vector bằng 0",
                "Ma trận basis có det bằng 0"
            ],
            "answer": 1,
            "explain": "Orthonormal = orthogonal + normalized."
        },
        {
            "q": "19. Numerical gradient có đặc điểm nào?",
            "options": [
                "Nhanh và exact",
                "Chậm, xấp xỉ nhưng dễ cài đặt",
                "Không thể dùng để kiểm tra code",
                "Không cần loss"
            ],
            "answer": 1,
            "explain": "Numerical gradient thường được dùng để gradient-check analytic gradient."
        },
        {
            "q": "20. 'No Free Lunch' trong ML nhấn mạnh điều gì?",
            "options": [
                "Luôn dùng neural network",
                "Luôn dùng linear regression",
                "Không có mô hình tốt nhất cho mọi bài toán",
                "Mọi mô hình đều cho kết quả giống nhau"
            ],
            "answer": 2,
            "explain": "Không có một mô hình duy nhất tối ưu trên mọi loại bài toán."
        },
    ]

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    with st.form("cs115_quiz"):
        responses = []
        for i, item in enumerate(quiz):
            choice = st.radio(
                item["q"],
                item["options"],
                index=None,
                key=f"quiz_{i}",
            )
            responses.append(choice)
        submitted = st.form_submit_button("Chấm điểm", use_container_width=True)

    if submitted:
        st.session_state.quiz_submitted = True
        score = 0
        unanswered = 0
        for i, item in enumerate(quiz):
            selected = st.session_state.get(f"quiz_{i}")
            if selected is None:
                unanswered += 1
            elif selected == item["options"][item["answer"]]:
                score += 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Điểm", f"{score}/{len(quiz)}")
        c2.metric("Tỷ lệ đúng", f"{score/len(quiz)*100:.0f}%")
        c3.metric("Chưa trả lời", unanswered)

        if score >= 17:
            st.success("Rất tốt — nền tảng khá chắc.")
        elif score >= 13:
            st.info("Khá tốt — nên xem lại các câu sai.")
        else:
            st.warning("Nên ôn lại phần công thức và các khái niệm nền tảng.")

        with st.expander("Xem đáp án và giải thích", expanded=True):
            for i, item in enumerate(quiz):
                selected = st.session_state.get(f"quiz_{i}")
                correct = item["options"][item["answer"]]
                if selected == correct:
                    st.markdown(f"**Câu {i+1}: ✅ Đúng** — {correct}")
                else:
                    shown = selected if selected is not None else "Chưa trả lời"
                    st.markdown(f"**Câu {i+1}: ❌ {shown}**  \nĐáp án: **{correct}**")
                st.caption(item["explain"])
                st.divider()

elif topic == "Bài tập có đáp án":
    st.header("✍️ Bài tập có đáp án")
    st.caption("Bài tập ngắn bám theo các nhóm kiến thức chính trong bộ tài liệu.")

    exercises = [
        (
            "Đại số tuyến tính — Dot product",
            "Cho x = (1, 2, -1), y = (2, 0, 3). Tính xᵀy và cho biết hai vector có trực giao không.",
            r"x^Ty = 1\cdot2 + 2\cdot0 + (-1)\cdot3 = -1.",
            "Vì xᵀy = -1 ≠ 0 nên hai vector không trực giao."
        ),
        (
            "Hình học giải tích — Norm và khoảng cách",
            "Cho x = (3,4), y = (0,0). Tính ||x||₂ và d(x,y).",
            r"\|x\|_2=\sqrt{3^2+4^2}=5,\qquad d(x,y)=\|x-y\|_2=5.",
            "Khi y là vector 0, khoảng cách từ x tới y chính là norm của x."
        ),
        (
            "Projection",
            "Cho v = (3,4), u = (1,0). Tính projection của v lên u.",
            r"\mathrm{proj}_u(v)=\frac{u^Tv}{u^Tu}u=\frac{3}{1}(1,0)=(3,0).",
            "Projection giữ lại thành phần của v theo hướng u."
        ),
        (
            "Xác suất — Binomial",
            "Một phép thử có xác suất thành công p=0.6. Thực hiện n=5 lần độc lập. Tính P(X=3).",
            r"P(X=3)=\binom53(0.6)^3(0.4)^2=10\times0.216\times0.16=0.3456.",
            "Đây là công thức phân phối nhị thức."
        ),
        (
            "Giải tích — Gradient",
            "Cho f(x,y)=x²+3y². Tính gradient tại (2,-1).",
            r"\nabla f=(2x,6y)\Rightarrow \nabla f(2,-1)=(4,-6).",
            "Negative gradient (-4,6) là hướng giảm nhanh nhất cục bộ."
        ),
        (
            "Linear Regression — Dự đoán",
            "Cho mô hình ŷ = b + wx với b=1.5, w=2 và x=3. Tính ŷ.",
            r"\hat y=1.5+2\cdot3=7.5.",
            "Đây là forward prediction của hồi quy tuyến tính 1 biến."
        ),
        (
            "Squared Error",
            "Nếu y=10 và ŷ=7.5, tính 1/2(y-ŷ)².",
            r"J=\frac12(10-7.5)^2=\frac12(2.5)^2=3.125.",
            "Hệ số 1/2 thường giúp đạo hàm gọn hơn."
        ),
        (
            "Gradient Descent",
            "Cho θ=5, gradient = 4, learning rate η=0.1. Tính θ mới.",
            r"\theta_{new}=5-0.1\cdot4=4.6.",
            "Gradient dương nên cập nhật theo hướng giảm làm θ nhỏ đi."
        ),
        (
            "L2 Regularization",
            "Cho w=(3,4), λ=0.1. Phần phạt L2 λ||w||² bằng bao nhiêu?",
            r"\lambda\|w\|_2^2=0.1(3^2+4^2)=0.1(25)=2.5.",
            "L2 phạt các trọng số lớn."
        ),
        (
            "Backpropagation — Chain rule",
            "Cho y=u² và u=3x. Tính dy/dx tại x=2.",
            r"\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}=2u\cdot3=6u.",
            "Tại x=2 ⇒ u=6 ⇒ dy/dx=36."
        ),
    ]

    for idx, (title, question, solution, note) in enumerate(exercises, start=1):
        st.subheader(f"Bài {idx}. {title}")
        st.markdown(question)
        with st.expander("Hiện đáp án"):
            st.latex(solution)
            st.info(note)

elif topic == "Visualization: Linear Regression":
    st.header("📈 Visualization — Linear Regression")
    st.caption("Tạo dữ liệu giả, fit đường thẳng bằng Gradient Descent và quan sát ảnh hưởng của noise/learning rate.")

    c1, c2, c3, c4 = st.columns(4)
    n = c1.slider("Số điểm dữ liệu", 20, 300, 80, 10)
    noise = c2.slider("Noise", 0.0, 10.0, 2.0, 0.5)
    lr = c3.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.05, 0.1], value=0.03)
    epochs = c4.slider("Epochs", 10, 500, 120, 10)

    true_w = st.slider("True slope w", -5.0, 5.0, 2.5, 0.1)
    true_b = st.slider("True bias b", -10.0, 10.0, 1.0, 0.5)
    seed = st.number_input("Random seed", value=42, step=1)

    rng = np.random.default_rng(int(seed))
    X = rng.uniform(-5, 5, int(n))
    y = true_w * X + true_b + rng.normal(0, noise, int(n))

    # Standardize X for stable GD while keeping predictions interpretable
    x_mean = X.mean()
    x_std = X.std() if X.std() > 1e-12 else 1.0
    Xs = (X - x_mean) / x_std

    w, b = 0.0, 0.0
    costs = []
    for _ in range(int(epochs)):
        y_hat = w * Xs + b
        err = y_hat - y
        cost = np.mean(err ** 2) / 2
        costs.append(cost)
        dw = np.mean(err * Xs)
        db = np.mean(err)
        w -= lr * dw
        b -= lr * db

    # convert standardized-space parameters back to original x-space
    fitted_w = w / x_std
    fitted_b = b - (w * x_mean / x_std)

    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = fitted_w * x_line + fitted_b

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(X, y, alpha=0.7, label="Data")
    ax.plot(x_line, y_line, linewidth=2, label="Fitted line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear Regression fitted by Gradient Descent")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated w", f"{fitted_w:.4f}", delta=f"{fitted_w-true_w:+.4f}")
    m2.metric("Estimated b", f"{fitted_b:.4f}", delta=f"{fitted_b-true_b:+.4f}")
    m3.metric("Final cost", f"{costs[-1]:.6f}")

    fig2, ax2 = plt.subplots(figsize=(8, 3.8))
    ax2.plot(range(1, len(costs) + 1), costs)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cost")
    ax2.set_title("Cost over epochs")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2, clear_figure=True)

    with st.expander("Giải thích thuật toán"):
        st.latex(r"\hat y_i = wx_i+b")
        st.latex(r"J(w,b)=\frac{1}{2N}\sum_{i=1}^{N}(\hat y_i-y_i)^2")
        st.latex(r"w\leftarrow w-\eta\frac{\partial J}{\partial w},\qquad b\leftarrow b-\eta\frac{\partial J}{\partial b}")
        st.markdown("""
- **Noise tăng** → dữ liệu khó fit hơn.
- **Learning rate quá nhỏ** → hội tụ chậm.
- **Learning rate quá lớn** → có thể dao động hoặc phân kỳ.
- **Epoch tăng** → có thêm bước cập nhật để tối ưu cost.
""")

elif topic == "Visualization: Gradient Descent":
    st.header("⛰️ Visualization — Gradient Descent")
    st.caption("Quan sát cách Gradient Descent tìm cực tiểu của một hàm bậc hai 1 chiều.")

    st.latex(r"f(x)=a(x-c)^2+d")

    c1, c2, c3 = st.columns(3)
    a = c1.slider("a (>0)", 0.1, 5.0, 1.0, 0.1)
    c = c2.slider("Vị trí cực tiểu c", -5.0, 5.0, 1.0, 0.1)
    d = c3.slider("Giá trị dịch d", -5.0, 5.0, 0.0, 0.5)

    c4, c5, c6 = st.columns(3)
    x0 = c4.slider("Điểm bắt đầu x₀", -10.0, 10.0, -7.0, 0.5)
    eta = c5.select_slider("Learning rate η", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8], value=0.1)
    steps = c6.slider("Số bước", 1, 50, 12)

    def f(x):
        return a * (x - c) ** 2 + d

    def grad(x):
        return 2 * a * (x - c)

    xs = [float(x0)]
    for _ in range(int(steps)):
        x_new = xs[-1] - eta * grad(xs[-1])
        if not np.isfinite(x_new) or abs(x_new) > 1e6:
            break
        xs.append(float(x_new))

    plot_min = min(-10, min(xs) - 1, c - 5)
    plot_max = max(10, max(xs) + 1, c + 5)
    xgrid = np.linspace(plot_min, plot_max, 500)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(xgrid, f(xgrid), label="f(x)")
    ax.scatter(xs, [f(x) for x in xs], zorder=3, label="GD steps")
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], f(xs[i + 1])),
            xytext=(xs[i], f(xs[i])),
            arrowprops=dict(arrowstyle="->", alpha=0.55),
        )
    ax.axvline(c, linestyle="--", alpha=0.7, label="Minimum x=c")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Gradient Descent trajectory")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)

    final_x = xs[-1]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("x cuối", f"{final_x:.6f}")
    mc2.metric("|x − c|", f"{abs(final_x-c):.6f}")
    mc3.metric("f(x cuối)", f"{f(final_x):.6f}")

    st.markdown("#### Bảng các bước cập nhật")
    rows = []
    for i, xval in enumerate(xs):
        rows.append({
            "step": i,
            "x": round(xval, 6),
            "f(x)": round(float(f(xval)), 6),
            "gradient": round(float(grad(xval)), 6),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if eta * 2 * a >= 2:
        st.warning(
            "Learning rate đang lớn đối với hàm này; Gradient Descent có thể dao động hoặc phân kỳ."
        )
    else:
        st.success("Thiết lập hiện tại nằm trong vùng thường hội tụ cho hàm bậc hai này.")

    with st.expander("Công thức"):
        st.latex(r"f'(x)=2a(x-c)")
        st.latex(r"x_{t+1}=x_t-\eta\,2a(x_t-c)")
        st.markdown("""
Nếu \(a>0\), cực tiểu thật nằm tại \(x=c\).  
Mục tiêu của Gradient Descent là khiến \(x_t\) tiến dần tới \(c\).
""")

st.divider()
st.caption("CS115 Review App — lý thuyết + quiz + bài tập + visualization.")

