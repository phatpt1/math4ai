import streamlit as st

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

st.divider()
st.caption("CS115 Review App — tổng hợp ngắn gọn để học và tra cứu trên Streamlit.")
