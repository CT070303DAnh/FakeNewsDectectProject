# CHI TIẾT LÝ THUYẾT 6 THUẬT TOÁN PHÂN LOẠI

## 1. Hồi quy Logistic (Logistic Regression)

### 🟢 Khái niệm
Hồi quy Logistic là thuật toán phân loại dùng để dự đoán xác suất một đối tượng thuộc về một nhóm cụ thể (thường là 0 hoặc 1). Thay vì vẽ một đường thẳng đi qua các điểm dữ liệu như Hồi quy Tuyến tính, nó sử dụng một hàm đặc biệt để "nén" kết quả vào khoảng [0, 1].

### 📐 Công thức & Chú thích
**Hàm dự đoán (Sigmoid):**
$$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Trong đó:
*   **$z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$**: Là tổng trọng số của các đặc trưng (Linear Score).
*   **$x_i$**: Các biến đầu vào (ví dụ: tần suất từ ngữ trong tin tức).
*   **$w_i$**: Trọng số (độ quan trọng) của từng đặc trưng. $w_0$ là sai số (bias).
*   **$e$**: Số Euler (~2.718).
*   **$P(y=1|x)$**: Xác suất đối tượng thuộc lớp 1 (ví dụ: Tin giả).

### ⚙️ Nguyên lý hoạt động
1.  **Tính điểm số ($z$):** Kết hợp các đặc trưng đầu vào với trọng số tương ứng.
2.  **Kích hoạt qua hàm Sigmoid:** Chuyển đổi điểm số $z$ (có thể từ âm vô cực đến dương vô cực) thành giá trị xác suất từ 0 đến 1.
3.  **Đưa ra quyết định:** So sánh xác suất với ngưỡng (thường là 0.5):
    *   Nếu $P \ge 0.5 \rightarrow$ Lớp 1 (Tin giả).
    *   Nếu $P < 0.5 \rightarrow$ Lớp 0 (Tin thật).

### 🎮 Mô phỏng tương tác
*   **Thanh trượt Trọng số ($w$):** Khi tăng $w$, độ dốc của hàm Sigmoid tăng lên, làm cho ranh giới phân loại trở nên "gắt" hơn.
*   **Điểm dữ liệu:** Di chuyển một điểm dọc theo trục $x$, người dùng sẽ thấy giá trị xác suất $P$ thay đổi mượt mà trên đường cong chữ S.

### ⚖️ Ưu & Nhược điểm
*   **Ưu điểm:** Đơn giản, tốc độ tính toán cực nhanh, cho biết xác suất cụ thể thay vì chỉ nhãn.
*   **Nhược điểm:** Chỉ hoạt động tốt khi ranh giới giữa 2 lớp là đường thẳng (tuyến tính); dễ bị ảnh hưởng bởi các biến có tương quan mạnh với nhau.

---

## 2. Máy vectơ hỗ trợ (Support Vector Machine - SVM)

### 🟢 Khái niệm
SVM mục tiêu là tìm một "siêu phẳng" (hyperplane) để phân chia các lớp dữ liệu sao cho khoảng cách (Margin) từ các điểm dữ liệu gần nhất của mỗi lớp đến siêu phẳng đó là lớn nhất.

### 📐 Công thức & Chú thích
**Phương trình siêu phẳng:**
$$w^T x + b = 0$$
**Khoảng cách lề (Margin):**
$$M = \frac{2}{||w||}$$
Trong đó:
*   **$w$**: Vectơ pháp tuyến của siêu phẳng (quyết định hướng của đường phân cách).
*   **$b$**: Hệ số tự do (định vị vị trí của đường phân cách).
*   **Vectơ hỗ trợ (Support Vectors)**: Những điểm dữ liệu nằm sát đường biên nhất. Chúng là những điểm "quan trọng nhất" quyết định hình dáng của SVM.

### ⚙️ Nguyên lý hoạt động
1.  **Xác định các điểm biên:** Tìm các điểm khó phân loại nhất của mỗi lớp (nằm gần lớp kia nhất).
2.  **Tối ưu hóa lề:** Tính toán sao cho khoảng trống giữa hai lớp là rộng nhất có thể.
3.  **Xử lý phi tuyến (Kernel):** Nếu dữ liệu không thể chia bằng đường thẳng, SVM dùng "Kernel Trick" để đưa dữ liệu lên không gian cao chiều hơn, nơi chúng có thể bị chia cắt dễ dàng.

### 🎮 Mô phỏng tương tác
*   **Kéo thả điểm:** Khi người dùng di chuyển các điểm không phải là Support Vector, đường biên sẽ không đổi. Khi di chuyển Support Vector, đường biên sẽ thay đổi theo.
*   **Chọn Kernel:** Chuyển đổi giữa "Linear" (đường thẳng) và "RBF" (vòng tròn/uốn lượn).

### ⚖️ Ưu & Nhược điểm
*   **Ưu điểm:** Rất hiệu quả trong không gian nhiều chiều; hoạt động tốt ngay cả khi số lượng đặc trưng lớn hơn số lượng mẫu.
*   **Nhược điểm:** Không phù hợp cho tập dữ liệu cực lớn vì thời gian huấn luyện lâu; khó giải thích kết quả (black-box).

---

## 3. Cây quyết định (Decision Tree)

### 🟢 Khái niệm
Cây quyết định mô phỏng cách con người đưa ra lựa chọn. Nó chia dữ liệu thành các nhóm nhỏ hơn dựa trên các câu hỏi "Đúng/Sai" về đặc trưng của dữ liệu, cho đến khi mỗi nhóm chỉ chứa các đối tượng cùng loại.

### 📐 Công thức & Chú thích
**Độ hỗn loạn (Entropy):**
$$H(S) = -\sum p_i \log_2(p_i)$$
**Chỉ số Gini (Độ tinh khiết):**
$$G = 1 - \sum p_i^2$$
Trong đó:
*   **$p_i$**: Tỉ lệ các mẫu thuộc lớp $i$ trong một nút.
*   **Thông tin thu được (Information Gain)**: Sự sụt giảm của Entropy sau khi chia dữ liệu. Cây luôn chọn câu hỏi nào mang lại Information Gain cao nhất.

### ⚙️ Nguyên lý hoạt động
1.  **Bắt đầu tại gốc:** Xem xét toàn bộ dữ liệu.
2.  **Chọn câu hỏi tốt nhất:** Thử tất cả các đặc trưng và chọn đặc trưng nào giúp phân loại dữ liệu "sạch" nhất (Gini hoặc Entropy thấp nhất).
3.  **Rẽ nhánh:** Chia dữ liệu thành các nhánh con và lặp lại bước 2 cho từng nhánh.
4.  **Dừng lại:** Khi tất cả dữ liệu trong nhánh cùng một lớp hoặc đạt giới hạn độ sâu (max depth).

### 🎮 Mô phỏng tương tác
*   **Nút bấm "Split":** Người dùng tự chọn đặc trưng để chia và xem Entropy giảm đi bao nhiêu.
*   **Độ sâu tối đa:** Điều chỉnh thanh trượt để thấy cây trở nên phức tạp và bắt đầu "học vẹt" (overfitting) như thế nào.

### ⚖️ Ưu & Nhược điểm
*   **Ưu điểm:** Cực kỳ dễ hiểu (có thể vẽ ra giấy), không cần chuẩn hóa dữ liệu.
*   **Nhược điểm:** Dễ bị "quá khớp" (overfitting) - máy nhớ máy móc dữ liệu cũ mà không dự đoán tốt dữ liệu mới.

---

## 4. Naive Bayes

### 🟢 Khái niệm
Đây là thuật toán dựa trên xác suất thống kê. Chữ "Naive" (Ngây ngô) ám chỉ việc thuật toán giả định rằng tất cả các đặc trưng đều độc lập với nhau (ví dụ: trong tin tức, sự xuất hiện của từ "tiền" không liên quan đến từ "trúng thưởng"), dù thực tế chúng thường đi cùng nhau.

### 📐 Công thức & Chú thích
**Định lý Bayes:**
$$P(C|x) = \frac{P(x|C) \times P(C)}{P(x)}$$
Trong đó:
*   **$P(C|x)$**: Xác suất hậu nghiệm (Sau khi thấy nội dung $x$, tỉ lệ là tin giả $C$ là bao nhiêu?).
*   **$P(x|C)$**: Khả năng xảy ra (Nếu là tin giả, xác suất xuất hiện các từ này là bao nhiêu?).
*   **$P(C)$**: Xác suất tiên nghiệm (Tỉ lệ tin giả nói chung trong hệ thống).
*   **$P(x)$**: Xác suất của dữ liệu (Hằng số chuẩn hóa).

### ⚙️ Nguyên lý hoạt động
1.  **Học bảng tần suất:** Thống kê xem mỗi từ xuất hiện bao nhiêu lần trong tin thật và tin giả.
2.  **Tính toán cho mẫu mới:** Khi có một bài báo mới, nhân tất cả xác suất của từng từ có trong bài báo đó với nhau cho từng lớp (Thật/Giả).
3.  **So sánh:** Lớp nào có kết quả tích xác suất cao hơn thì chọn lớp đó.

### 🎮 Mô phỏng tương tác
*   **Nhập từ khóa:** Người dùng nhập các từ như "Khuyến mãi", "Sự thật". Hệ thống hiển thị các cột xác suất nhảy lên xuống tương ứng với từng lớp.
*   **Bật/Tắt giả định độc lập:** Giải thích tại sao việc nhân các xác suất lại đơn giản hóa bài toán.

### ⚖️ Ưu & Nhược điểm
*   **Ưu điểm:** Cực kỳ nhanh, hiệu quả vượt trội với bài toán phân loại văn bản (Spam, Tin tức).
*   **Nhược điểm:** Giả định độc lập thường sai trong thực tế; nếu một từ chưa bao giờ xuất hiện trong lúc học, xác suất sẽ bằng 0 và làm hỏng toàn bộ phép nhân (cần kỹ thuật Laplace smoothing).

---

## 5. Rừng ngẫu nhiên (Random Forest)

### 🟢 Khái niệm
Thay vì tin vào một Cây quyết định, Random Forest xây dựng hàng trăm cây khác nhau và cho chúng "bầu chọn". Đây là một kỹ thuật thuộc nhóm Ensemble Learning (Học kết hợp).

### 📐 Công thức & Chú thích
**Kết quả cuối cùng:**
$$\hat{y} = \text{mode}\{T_1(x), T_2(x), ..., T_n(x)\}$$
Trong đó:
*   **$T_i(x)$**: Dự đoán của cây thứ $i$.
*   **$\text{mode}$**: Giá trị xuất hiện nhiều nhất (Bầu chọn theo đa số).
*   **Bootstrap**: Mỗi cây chỉ được học một phần dữ liệu được lấy ngẫu nhiên.

### ⚙️ Nguyên lý hoạt động
1.  **Lấy mẫu ngẫu nhiên (Bagging):** Tạo ra nhiều tập dữ liệu con từ tập gốc bằng cách lấy mẫu có lặp lại.
2.  **Trồng rừng:** Trên mỗi tập con, huấn luyện một Cây quyết định. Điểm đặc biệt là tại mỗi nút chia, cây cũng chỉ được chọn ngẫu nhiên một vài đặc trưng để xem xét.
3.  **Hợp lực:** Khi cần dự đoán, đưa dữ liệu qua tất cả các cây.
4.  **Bầu chọn:** Thu thập kết quả từ tất cả các cây, nhãn nào được chọn nhiều nhất sẽ là kết quả cuối cùng.

### 🎮 Mô phỏng tương tác
*   **Số lượng cây:** Tăng số cây và xem đường biên phân loại trở nên ổn định và bớt răng cưa hơn.
*   **Xem từng cây:** Click để xem cấu trúc của 3-4 cây ngẫu nhiên trong rừng để thấy chúng khác nhau như thế nào.

### ⚖️ Ưu & Nhược điểm
*   **Ưu điểm:** Độ chính xác rất cao, hạn chế tối đa việc quá khớp, xử lý được dữ liệu thiếu.
*   **Nhược điểm:** Tốn tài nguyên (RAM/CPU) hơn cây đơn lẻ; khó giải thích vì có quá nhiều cây.

---

## 6. PhoBERT (Transformer cho tiếng Việt)

### 🟢 Khái niệm
PhoBERT là một mô hình ngôn ngữ dựa trên kiến trúc Transformer (như ChatGPT nhưng chuyên biệt hơn), được huấn luyện trên khối lượng dữ liệu khổng lồ tiếng Việt. Khác với các mô hình truyền thống chỉ đếm từ, PhoBERT hiểu được ngữ cảnh sâu sắc của câu văn thông qua cơ chế học sâu.

### 📐 Công thức & Chú thích
**Cơ chế Chú ý (Self-Attention):**
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Trong đó:
*   **$Q, K, V$**: Các ma trận Query (Truy vấn), Key (Khóa) và Value (Giá trị) được tạo ra từ văn bản đầu vào.
*   **$\text{softmax}$**: Hàm chuẩn hóa xác suất để biết từ nào nên "chú ý" vào từ nào nhất.
*   **$\sqrt{d_k}$**: Yếu tố tỉ lệ để giữ cho các giá trị không quá lớn.
*   **Transformer Layers**: Các lớp mạng nơ-ron chồng lên nhau để trích xuất ý nghĩa từ thấp đến cao (từ chữ cái -> từ -> ngữ pháp -> ý nghĩa).

### ⚙️ Nguyên lý hoạt động
1.  **Phân đoạn từ (Segmentation):** Sử dụng các thư viện như `underthesea` để tách từ tiếng Việt đúng cách (ví dụ: "học sinh" là 1 từ, không phải 2 chữ "học" và "sinh").
2.  **Mã hóa (Tokenization):** Chuyển từ thành các con số định danh (IDs) mà máy tính hiểu được.
3.  **Cơ chế Chú ý:** Khi đọc một từ, mô hình sẽ nhìn các từ xung quanh để xác định nghĩa chính xác dựa trên ngữ cảnh (Ví dụ: từ "đường" trong "đường ăn" khác với "đường đi").
4.  **Phân loại (Classification head):** Một lớp mạng nơ-ron cuối cùng sẽ nhận toàn bộ thông tin ngữ cảnh để đưa ra kết luận: Tin thật hay Tin giả.

### 🎮 Mô phỏng tương tác
*   **Ma trận Attention:** Khi người dùng di chuột qua một từ, các từ liên quan trong câu sẽ sáng lên (thể hiện mức độ liên kết ngữ nghĩa).
*   **Lớp Transformer:** Hiển thị cách văn bản biến đổi qua từng lớp nơ-ron từ dữ liệu thô thành các vector trừu tượng.

### ⚖️ Ưu & Nhược điểm
*   **Ưu điểm:** Độ chính xác cực cao, hiểu được sắc thái biểu cảm, tiếng lóng và ngữ pháp phức tạp của tiếng Việt.
*   **Nhược điểm:** Cần cấu hình máy tính mạnh (nên có GPU), tốc độ dự đoán chậm hơn các mô hình truyền thống; khó giải thích nội tại vì có hàng triệu tham số.

