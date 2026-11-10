# BẢNG SO SÁNH CHI TIẾT 8 MÔ HÌNH PHÂN LOẠI TIN GIẢ

Bảng dưới đây so sánh 8 mô hình được triển khai trong hệ thống, bao gồm 5 mô hình Machine Learning truyền thống và 3 mô hình Deep Learning tiên tiến (Transformers).

| Tiêu chí | Logistic Regression | Support Vector Machine (SVM) | Decision Tree | Naive Bayes | Random Forest | PhoBERT | ViBERT | SBERT (Vietnamese) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Khái niệm** | Thuật toán phân loại dựa trên xác suất, sử dụng hàm Sigmoid để đưa ra kết quả 0 hoặc 1. | Tìm siêu phẳng tối ưu để phân tách dữ liệu với khoảng cách lề (Margin) lớn nhất. | Cấu trúc phân nhánh dạng cây, đưa ra quyết định dựa trên các câu hỏi Đúng/Sai. | Dựa trên định lý Bayes với giả định các đặc trưng độc lập với nhau. | Tập hợp (Ensemble) của nhiều cây quyết định để bầu chọn kết quả cuối cùng. | Mô hình Transformer (BERT) được huấn luyện chuyên biệt cho tiếng Việt bởi VinAI. | Biến thể BERT cho tiếng Việt được phát triển bởi FPT AI. | Sentence-BERT tối ưu hóa việc tạo vector biểu diễn cho toàn bộ câu văn. |
| **2. Nguyên lý hoạt động** | Tính tổng trọng số của các từ khóa (TF-IDF) rồi đi qua hàm kích hoạt Sigmoid. | Sử dụng các điểm dữ liệu biên (Support Vectors) để xác định ranh giới phân loại. | Chia nhỏ dữ liệu dựa trên chỉ số Entropy hoặc Gini để đạt độ tinh khiết cao nhất. | Tính xác suất hậu nghiệm dựa trên tần suất xuất hiện của từ ngữ trong tập tin thật/giả. | Sử dụng kỹ thuật Bagging và lấy mẫu ngẫu nhiên để giảm thiểu hiện tượng quá khớp (overfitting). | Cơ chế Self-Attention giúp hiểu mối quan hệ giữa các từ trong câu theo cả hai chiều. | Sử dụng kiến trúc BERT-base cased, tập trung vào các đặc trưng ngữ pháp và thực thể. | Sử dụng kiến trúc Siamese Network để học các câu có ý nghĩa tương đương trong không gian vector. |
| **3. Luồng xử lý** | Preprocess -> TF-IDF -> Logistic Regression | Preprocess -> TF-IDF -> SVM Kernel | Preprocess -> TF-IDF -> Cây quyết định | Preprocess -> TF-IDF -> Multinomial NB | Preprocess -> TF-IDF -> Random Forest | Preprocess -> Tokenization -> Transformer Layers -> Classification Head | Preprocess -> Tokenization (cased) -> BERT Layers -> Classification Head | Preprocess -> Tokenization -> Sentence Embedding -> Classification Head |
| **4. Ưu điểm** | Tốc độ cực nhanh, dễ giải thích trọng số của từng từ khóa. | Hiệu quả trong không gian nhiều chiều, độ chính xác ổn định. | Trực quan, dễ hiểu, không cần chuẩn hóa dữ liệu phức tạp. | Cực nhanh, hiệu quả với dữ liệu văn bản ít từ hoặc tập dữ liệu nhỏ. | Độ chính xác cao, bền vững, ít bị nhiễu bởi các điểm dữ liệu lạ. | Hiểu ngữ cảnh tiếng Việt sâu sắc, xử lý tốt từ lóng và ngữ nghĩa phức tạp. | Hiệu quả với văn bản tiếng Việt có cấu trúc chính quy, báo chí. | Tạo ra vector câu chất lượng cao, nắm bắt ý nghĩa toàn cục của bài viết. |
| **5. Nhược điểm** | Khó xử lý các mối quan hệ phi tuyến phức tạp giữa các từ. | Thời gian huấn luyện lâu khi dữ liệu lớn, khó giải thích kết quả. | Dễ bị quá khớp (học vẹt) nếu cây quá sâu. | Giả định "độc lập" thường không đúng trong ngôn ngữ tự nhiên. | Tốn tài nguyên RAM, thời gian chạy lâu hơn cây đơn lẻ. | Đòi hỏi GPU mạnh, tốc độ dự đoán (Inference) chậm hơn ML truyền thống. | Tương tự PhoBERT, đòi hỏi tài nguyên tính toán cao. | Phức tạp trong việc tinh chỉnh (fine-tuning) cho các tác vụ cụ thể. |
| **7. Ứng dụng hiệu quả nhất** | Phân loại nhanh các tin tức ngắn, rõ ràng về từ vựng. | Khi cần độ chính xác cao trên tập dữ liệu trung bình. | Dùng làm mô hình cơ sở để hiểu các quy tắc phân loại đơn giản. | Phân loại rác (Spam) hoặc tin tức dựa trên từ khóa đặc trưng. | Khi cần một mô hình ML mạnh mẽ và ổn định trên mọi loại dữ liệu. | **Tin giả tinh vi, ẩn dụ, tin tức mạng xã hội có ngữ cảnh phức tạp.** | **Tin tức báo chí chính thống, văn bản có tính học thuật cao.** | **So sánh sự tương đồng giữa các tin tức, phát hiện tin giả sao chép.** |

---

## Phân tích chuyên sâu về hiệu quả ứng dụng

1.  **Nhóm Truyền thống (Logistic, SVM, NB, DT, RF):**
    *   **Hiệu quả nhất:** Khi bạn cần một hệ thống phản hồi tức thì (Real-time) và tài nguyên phần cứng hạn chế. 
    *   **Trường hợp lý tưởng:** Các bài báo có xu hướng lặp đi lặp lại các từ khóa "lừa đảo", "trúng thưởng", "khẩn cấp" một cách rõ ràng.

2.  **Nhóm PhoBERT & ViBERT:**
    *   **Hiệu quả nhất:** Phân tích các bài đăng trên mạng xã hội (Facebook, TikTok) nơi câu văn không đầy đủ hoặc sử dụng từ ngữ đa nghĩa.
    *   **Điểm mạnh:** Có khả năng phân biệt từ "đường" trong "đường ăn" và "đường đi" nhờ cơ chế Attention, điều mà TF-IDF hoàn toàn thất bại.

3.  **Nhóm SBERT:**
    *   **Hiệu quả nhất:** Khi cần phát hiện một tin giả được xào nấu lại từ một tin thật. SBERT sẽ nhận diện được ý nghĩa của hai câu văn là tương đồng dù cách dùng từ có thể khác nhau.

---
*Bảng so sánh này được tổng hợp dựa trên cấu trúc thư mục `models/` và mã nguồn trong `src/` của dự án.*
