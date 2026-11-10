# BAN CƠ YẾU CHÍNH PHỦ
# HỌC VIỆN KỸ THUẬT MẬT MÃ
¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯

## ĐỀ CƯƠNG ĐỒ ÁN TỐT NGHIỆP
**Đề tài (STT 8): Xây dựng hệ thống AI hỗ trợ kiểm chứng thông tin tiếng Việt dựa trên truy xuất và phân tích bằng chứng**

* **Ngành:** Công nghệ thông tin
* **Mã số:** 748.02.01
* **Sinh viên thực hiện:** Bùi Đức Anh
* **Lớp:** CT...
* **Người hướng dẫn:** ThS. Thái Thị Thanh Vân — Học viện Kỹ thuật Mật mã

*Hà Nội, 2026*

---

## I. MỞ ĐẦU

### 1. Tính cấp thiết của đề tài
Sự bùng nổ của mạng xã hội và các nền tảng truyền thông số tại Việt Nam đã làm thay đổi căn bản cách thức thông tin được tạo lập, chia sẻ và tiếp nhận. Bên cạnh những lợi ích về tốc độ truyền tải, môi trường mạng cũng trở thành không gian phát tán nhanh chóng của tin giả (fake news), tin đồn thất thiệt và các kịch bản lừa đảo trực tuyến trong nhiều lĩnh vực nhạy cảm như y tế, tài chính – ngân hàng, an ninh trật tự và chính sách công. Theo nghiên cứu quy mô lớn đăng trên tạp chí *Science* của Vosoughi, Roy và Aral (2018), thông tin sai lệch trên mạng xã hội có tốc độ lan truyền nhanh hơn, sâu hơn và rộng hơn gấp 6 lần so với tin thật do đặc tính mới lạ (novelty) và khả năng kích động cảm xúc mạnh (surprise, fear, disgust) của người đọc.

Các phương pháp phát hiện tin giả truyền thống thường tiếp cận dưới góc độ phân loại văn bản thuần túy (Text Classification) dựa trên đặc trưng từ vựng hoặc văn phong. Tuy nhiên, hướng tiếp cận này bộc lộ hạn chế lớn trong thực tiễn: các mô hình học máy dễ bị đánh lừa bởi những tin giả được ngụy trang dưới văn phong báo chí trang trọng, hoặc ngược lại, nhận diện nhầm (False Positive) các bài báo chính thống đang trích dẫn lại tin đồn để đính chính/bác bỏ là tin giả. Đặc biệt, đối với các sự kiện mới phát sinh hoặc tin cũ bị gán ghép sai thời điểm, một mô hình phân loại tĩnh không thể xác định tính đúng/sai nếu không đối chiếu với nguồn bằng chứng xác thực và mốc thời gian xuất bản bên ngoài.

Xuất phát từ thực tế đó, xu hướng nghiên cứu hiện đại chuyển dịch sang **kiểm chứng thông tin dựa trên truy xuất và phân tích bằng chứng (Evidence-based Fact-Checking)**. Thay vì chỉ dự đoán cảm tính qua mặt chữ, hệ thống thực hiện phát hiện và phân loại câu đáng kiểm chứng (*Claim Detection & Check-worthiness*), chuẩn hóa mệnh đề (*Claim Normalization*), tự động truy xuất bằng chứng lai (*Hybrid Retrieval: BM25 + Dense Vector*) kết hợp xếp hạng lại (*Cross-Encoder Reranking & Temporal Filtering*), sau đó áp dụng suy luận ngôn ngữ tự nhiên (*Natural Language Inference – NLI*) và đối chứng với *LLM Few-shot Evidence-Grounded Verification* để đánh giá mệnh đề theo 3 mức: **Ủng hộ (`Supported`)**, **Bác bỏ (`Refuted`)** hoặc **Chưa đủ bằng chứng (`Not Enough Evidence - NEI`)**.

Đề tài **"Xây dựng hệ thống AI hỗ trợ kiểm chứng thông tin tiếng Việt dựa trên truy xuất và phân tích bằng chứng"** giải quyết trọn vẹn bài toán xác thực thông tin có căn cứ minh bạch, đồng thời kết hợp tín hiệu Độ mới lạ thông tin (*Information Uniqueness, KL-Divergence, Bhattacharyya Distance*) và Cảm xúc/Phản bác theo phương pháp luận của *Vosoughi et al. (2018)* để xử lý độ bất định (*Uncertainty Handling*) khi bằng chứng chưa đầy đủ. Qua đó, hệ thống cung cấp giải thích chi tiết (*Explanation*), trích dẫn nguồn (*Citation*), thời điểm bằng chứng (*Evidence Date*) và cơ chế thu thập phản hồi người dùng (*User Feedback*) phục vụ làm giàu kho tri thức kiểm chứng tiếng Việt.

### 2. Mục tiêu nghiên cứu của đề tài
- Xây dựng hệ thống AI xác định các phát biểu có thể kiểm chứng (*Claim Detection & Check-worthiness*), chuẩn hóa mệnh đề, truy xuất bằng chứng từ kho dữ liệu được kiểm soát kết hợp báo chí chính thống và đánh giá phát biểu theo 3 mức **`Supported` / `Refuted` / `Not Enough Evidence (NEI)`**.
- Thiết kế và thực nghiệm kiến trúc Truy xuất lai (**BM25 Baseline + Dense Semantic Retrieval + Cross-Encoder Reranker + Temporal Filtering**) kết hợp mô hình suy luận **NLI/Transformer** và đối sánh với **LLM Few-shot Evidence-Grounded Verification**.
- Xây dựng bộ dữ liệu đánh giá chuẩn **Vietnamese Fact Verification Evaluation Set (360 claims chất lượng cao)** gồm đầy đủ 6 trường: `claim`, `label`, `gold_evidence`, `source`, `date`, `topic` cùng kho bằng chứng (Evidence Corpus).
- Cung cấp giao diện Web trực quan hiển thị mức độ tự tin (`confidence`), độ bất định (`uncertainty`), lập luận giải thích (`explanation`), trích dẫn nguồn kèm thời gian (`citation & date`) và thu thập phản hồi người dùng (`user feedback`).

### 3. Đối tượng và phạm vi nghiên cứu
- **Đối tượng nghiên cứu:**
  + Bài toán phát hiện mệnh đề đáng kiểm chứng (*Check-worthiness Text Classification*), trích xuất và chuẩn hóa mệnh đề (*Claim Extraction & Normalization*).
  + Các kỹ thuật truy xuất và xếp hạng bằng chứng: **Okapi BM25 Baseline**, **Dense Retrieval** (`Sentence-BERT`/`SVD`), **Hybrid Retrieval (BM25 + Dense)**, **Cross-Encoder Reranker** và lọc theo siêu dữ liệu/thời gian (**Metadata & Temporal Filtering**).
  + Bài toán kiểm chứng dựa trên bằng chứng (*Evidence-grounded Verification*) phân loại 3 nhãn (`Supported`, `Refuted`, `NEI`): so sánh giữa mô hình **NLI/Transformer**, **LLM Few-shot Verification** và sự đồng thuận của 8 mô hình Học máy/Học sâu (`Logistic Regression`, `SVM`, `Random Forest`, `Naive Bayes`, `Decision Tree`, `PhoBERT`, `ViBERT`, `Vietnamese-SBERT`) cùng các độ đo Độ mới lạ thông tin ($IU, KL, BD$).
- **Phạm vi nghiên cứu:**
  + Kiểm chứng các bài viết, mệnh đề khẳng định hoặc đường dẫn URL bài báo tiếng Việt thuộc các chủ đề: Tài chính – Tiền tệ, Y tế – Sức khỏe, An ninh mạng – Pháp luật, Chính sách – Xã hội, Giáo dục – Tuyển sinh và Giao thông – Đô thị.
  + Truy xuất bằng chứng trên Kho dữ liệu kiểm chứng tiếng Việt được kiểm soát (**Vietnamese Fact Verification Corpus**) kết hợp nguồn báo chí chính thống trực tuyến (Google News RSS Tiếng Việt) và lưu trữ phản hồi người dùng qua `SQLite`/JSON.

### 4. Các nhiệm vụ chính cần thực hiện
Nội dung nghiên cứu của đồ án tập trung vào các nhiệm vụ chính sau:
- Nghiên cứu cơ sở lý thuyết về kiểm chứng thông tin tự động, xử lý ngôn ngữ tự nhiên tiếng Việt (`underthesea`, chuẩn hóa Unicode NFC/Teencode, NER) và phương pháp luận của *Vosoughi et al. (Science, 2018)*.
- Xây dựng bộ dữ liệu **Vietnamese Fact Verification Evaluation Set (360 claims, cân bằng 120 `Supported` – 120 `Refuted` – 120 `NEI`)** có đầy đủ `claim`, `label`, `gold_evidence`, `source`, `date`, `topic` và bộ kiểm thử `Check-worthiness` (300 mẫu).
- Thiết kế và cài đặt hệ thống theo kiến trúc **5 Giai đoạn (12 Node xử lý)** giữ nguyên luồng dữ liệu chuẩn:
  1. *Giai đoạn 1 (Nodes 1–3):* Tiếp nhận bài viết/URL, chuẩn hóa tiếng Việt, phân loại câu đáng kiểm chứng (*Check-worthiness Classification*) và chuẩn hóa Claim (*Claim Normalization*).
  2. *Giai đoạn 2 (Nodes 4–5):* Truy xuất và xếp hạng bằng chứng với **BM25 Baseline**, **Dense Retrieval**, **Hybrid Retrieval (BM25 + Dense)**, **Cross-Encoder Reranker** và lọc thời điểm bằng chứng (**Temporal Filtering** theo trường `date` & ngưỡng S2.2 `0.50 / 0.90`).
  3. *Giai đoạn 3 (Nodes 6–8):* Phân tích tín hiệu nội tại với 8 mô hình ML/DL, đo Độ mới lạ thông tin ($IU, KL, BD$), Cảm xúc/Phản bác và nhận diện bối cảnh đính chính (*Debunking Context*) phục vụ xử lý độ bất định (*Uncertainty Handling*).
  4. *Giai đoạn 4 (Nodes 9–11):* Suy luận quan hệ mệnh đề – bằng chứng theo 3 nhãn **`Supported` / `Refuted` / `NEI`**, đối sánh giữa **NLI Verifier** với **LLM Few-shot Evidence-Grounded Verification**, kiểm tra trích dẫn (*Citation*) và hợp nhất phán quyết.
  5. *Giai đoạn 5 (Node 12):* Xây dựng RESTful API (`FastAPI`) và Giao diện Web (`React/TypeScript`) hiển thị `confidence`, `explanation`, `citation`, `evidence date`, `latency` và thu thập **Feedback người dùng (`POST /feedback`)** lưu vào `SQLite`.
- Thực nghiệm và đánh giá toàn diện theo đúng **7 nhóm tiêu chí bắt buộc**:
  1. `Claim Detection Macro-F1`
  2. `Retrieval Recall@K (K=1,3,5)`, `MRR@5`, `NDCG@5`
  3. `Verification Macro-F1` (trên 3 nhãn `Supported` / `Refuted` / `NEI`)
  4. `Evidence Precision / Recall`
  5. `Citation Accuracy`
  6. `End-to-End Evidence-Grounded Accuracy`
  7. `Latency (ms)`

### 5. Kết quả dự kiến
- Hệ thống Web/API Prototype hoàn chỉnh hỗ trợ kiểm chứng thông tin tiếng Việt dựa trên truy xuất và phân tích bằng chứng (`FastAPI` + `React/TypeScript` + `SQLite`).
- Bộ dữ liệu **Vietnamese Fact Verification Evaluation Set (360 claims)** và **Evidence Corpus** chuẩn hóa có đầy đủ `claim`, `label`, `gold_evidence`, `source`, `date`, `topic`.
- Bộ mô hình Truy xuất (`BM25`, `Dense`, `Hybrid + Cross-Encoder Reranker`) và mô hình Kiểm chứng (`NLI Verifier`, `LLM Few-shot Grounded Verifier` + 8 mô hình phân loại ML/DL).
- Bộ kịch bản kiểm thử tự động (`src/test_pipeline.py`, `src/evaluate_benchmark.py`) và báo cáo thực nghiệm đo đầy đủ 7 nhóm chỉ số chuẩn (`Claim Detection Macro-F1`, `Recall@K/MRR/NDCG`, `Verification Macro-F1`, `Evidence P/R`, `Citation Accuracy`, `End-to-End Grounded Accuracy`, `Latency`).

---

## II. DỰ KIẾN CÁC CHƯƠNG MỤC

**MỤC LỤC**
**DANH MỤC CÁC TỪ VIẾT TẮT**
**DANH MỤC CÁC BẢNG BIỂU**
**DANH MỤC CÁC HÌNH VẼ**
**LỜI CẢM ƠN**
**MỞ ĐẦU**

**CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI VÀ CƠ SỞ LÝ THUYẾT**
* 1.1. Tổng quan về bài toán kiểm chứng thông tin dựa trên bằng chứng (Evidence-based Fact-Checking)
  * 1.1.1. Khái niệm, thách thức của tin giả tiếng Việt và hạn chế của phân loại văn bản thuần túy
  * 1.1.2. Quy trình kiểm chứng thông tin chuẩn: Claim Detection $\rightarrow$ Evidence Retrieval $\rightarrow$ NLI Verification
  * 1.1.3. Phương pháp luận từ nghiên cứu của Vosoughi et al. (Science, 2018) và cơ chế xử lý độ bất định (Uncertainty Handling)
* 1.2. Cơ sở lý thuyết về Truy xuất bằng chứng và Suy luận ngôn ngữ tự nhiên
  * 1.2.1. Bài toán phát hiện và chuẩn hóa mệnh đề đáng kiểm chứng (Claim Detection & Check-worthiness Classification)
  * 1.2.2. Kỹ thuật truy xuất từ khóa (Okapi BM25), truy xuất ngữ nghĩa đặc (Dense Retrieval) và truy xuất lai (Hybrid Retrieval)
  * 1.2.3. Kỹ thuật xếp hạng lại (Cross-Encoder Reranking) và lọc theo thời gian (Temporal Filtering)
  * 1.2.4. Mô hình suy luận quan hệ mệnh đề – bằng chứng 3 nhãn (`Supported`, `Refuted`, `NEI`) và LLM Few-shot Evidence-Grounded Verification
  * 1.2.5. Các mô hình Học máy, Học sâu tiếng Việt (`PhoBERT`, `ViBERT`, `Vietnamese-SBERT`) và độ đo Độ mới lạ ($IU, KL, BD$)
* 1.3. Công nghệ và công cụ sử dụng (`Python`, `FastAPI`, `React`, `TypeScript`, `Scikit-learn`, `PyTorch`, `SQLite`)
* 1.4. Tổng kết chương 1

**CHƯƠNG 2. PHÂN TÍCH, THIẾT KẾ HỆ THỐNG VÀ XÂY DỰNG BỘ DỮ LIỆU**
* 2.1. Phát biểu bài toán và đặc tả yêu cầu hệ thống
  * 2.1.1. Yêu cầu chức năng (11 chức năng cốt lõi từ tách Claim, Hybrid Retrieval, NLI Verification đến Citation & User Feedback)
  * 2.1.2. Yêu cầu phi chức năng (Độ trễ Latency, tính giải trình Explainability, độ chính xác Citation)
* 2.2. Xây dựng Kho bằng chứng (Evidence Corpus) và Bộ dữ liệu đánh giá (Vietnamese Fact Verification Evaluation Set – 360 Claims)
  * 2.2.1. Cấu trúc dữ liệu 6 trường chuẩn (`claim`, `label`, `gold_evidence`, `source`, `date`, `topic`)
  * 2.2.2. Quy trình thu thập, chuẩn hóa và gán nhãn 3 lớp (`Supported`, `Refuted`, `NEI`)
* 2.3. Thiết kế kiến trúc tổng thể hệ thống (Pipeline 5 Giai đoạn – 12 Node xử lý)
* 2.4. Biểu đồ Use Case, Biểu đồ Tuần tự (Sequence Diagrams) và Thiết kế Cơ sở dữ liệu (`SQLite` & `JSON Schema`)
* 2.5. Thiết kế thuật toán Hợp nhất phán quyết, Xử lý độ bất định (Uncertainty) và Sinh giải thích có trích dẫn (Grounded Explanation)
* 2.6. Tổng kết chương 2

**CHƯƠNG 3. XÂY DỰNG VÀ CÀI ĐẶT HỆ THỐNG**
* 3.1. Cài đặt Giai đoạn 1 (`src/preprocess.py`): Thu thập URL, chuẩn hóa tiếng Việt, phân loại Check-worthiness và chuẩn hóa Claim
* 3.2. Cài đặt Giai đoạn 2 (`src/evidence_retriever.py`): Chỉ mục Okapi BM25, Dense Vector, Hybrid Fusion, Cross-Encoder Reranker và Temporal Filtering
* 3.3. Cài đặt Giai đoạn 3 (`src/novelty_emotion.py`, `src/train.py`): Tính toán Độ mới lạ ($IU, KL, BD$), Cảm xúc, Bối cảnh bác bỏ và 8 mô hình phân loại
* 3.4. Cài đặt Giai đoạn 4 (`src/evidence_analyzer.py`): Mô-đun kiểm chứng 3 nhãn (`Supported`/`Refuted`/`NEI`), đối sánh NLI Verifier với LLM Few-shot Grounded Verification
* 3.5. Cài đặt Giai đoạn 5 (`main.py`, `frontend/src/App.tsx`, `TrainingPage.tsx`): RESTful API, giao diện hiển thị Confidence/Explanation/Citation/Date và chức năng User Feedback
* 3.6. Tổng kết chương 3

**CHƯƠNG 4. THỰC NGHIỆM, ĐÁNH GIÁ VÀ SO SÁNH KẾT QUẢ**
* 4.1. Môi trường thực nghiệm và thiết lập các độ đo chuẩn (`Macro-F1`, `Recall@K`, `MRR`, `NDCG`, `Citation Accuracy`, `Latency`)
* 4.2. Thực nghiệm Đánh giá Phát hiện Mệnh đề (Claim Detection & Check-worthiness Macro-F1)
* 4.3. Thực nghiệm So sánh các phương pháp Truy xuất Bằng chứng (`BM25 Baseline` vs. `Dense` vs. `Hybrid` vs. `Hybrid + Cross-Encoder Reranker + Temporal`)
* 4.4. Thực nghiệm So sánh Kiểm chứng 3 nhãn (`NLI Verifier` vs. `LLM Few-shot Grounded Verification` vs. `Hybrid Consensus`) và hiệu năng 8 mô hình ML/DL
* 4.5. Đánh giá Độ chính xác Toàn trình có neo bằng chứng (`End-to-End Evidence-Grounded Accuracy`), `Citation Accuracy` và Thời gian phản hồi (`Latency`)
* 4.6. Tổng kết chương 4

**KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
**TÀI LIỆU THAM KHẢO**
**PHỤ LỤC**

---

## III. TÀI LIỆU THAM KHẢO ĐỂ XÂY DỰNG ĐỀ CƯƠNG

[1] Vosoughi, S., Roy, D., & Aral, S. (2018). *The spread of true and false news online*. Science, 359(6380), 1146–1151. DOI: 10.1126/science.aap9559.
[2] Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification*. In Proceedings of NAACL-HLT 2018 (pp. 809–819).
[3] Guo, Z., Schlichtkrull, M., & Vlachos, A. (2022). *A Survey on Automated Fact-Checking*. Transactions of the Association for Computational Linguistics (TACL), 10, 178–206.
[4] Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389.
[5] Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese*. In Findings of EMNLP 2020 (pp. 1037–1042).
[6] Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of EMNLP 2019 (pp. 3982–3992).
[7] Le, D. T., Vu, X. S., & Nguyen, M. L. (2020). *ReINTEL: A Multimodal Data Challenge for Responsible Information Identification on Vietnamese Social Media*. In Proceedings of VLSP 2020.

---

## IV. KẾ HOẠCH THỰC HIỆN

| STT | Thời gian | Nội dung thực hiện | Kết quả dự kiến |
| :---: | :---: | :--- | :--- |
| 1 | Tuần 1 – Tuần 2 | Nghiên cứu quy trình kiểm chứng thông tin dựa trên bằng chứng (Claim Detection, Hybrid Retrieval, NLI Verification) và bài báo *Vosoughi et al. (Science, 2018)*. | Báo cáo tổng quan lý thuyết và đề cương chi tiết. |
| 2 | Tuần 3 – Tuần 4 | Xây dựng Kho bằng chứng (Evidence Corpus) và bộ dữ liệu **Vietnamese Fact Verification Evaluation Set (360 claims)** gồm `claim`, `label`, `gold_evidence`, `source`, `date`, `topic`. | Bộ dữ liệu đánh giá 360 claims chuẩn hóa và tập kiểm thử Check-worthiness. |
| 3 | Tuần 5 | Thiết kế kiến trúc 5 Giai đoạn (12 Node xử lý) và cài đặt Giai đoạn 1 (`preprocess.py`): Phát hiện mệnh đề (Check-worthiness), chuẩn hóa Claim và trích xuất thực thể/thời gian. | Module phát hiện và chuẩn hóa mệnh đề đạt Macro-F1 cao. |
| 4 | Tuần 6 – Tuần 7 | Cài đặt Giai đoạn 2 (`evidence_retriever.py`): Xây dựng `BM25 Baseline`, `Dense Retrieval`, `Hybrid Retrieval`, `Cross-Encoder Reranker` và bộ lọc thời gian (`Temporal Filtering`). | Module truy xuất và xếp hạng bằng chứng đa chế độ. |
| 5 | Tuần 8 – Tuần 9 | Cài đặt Giai đoạn 3 (`novelty_emotion.py`, `train.py`): Huấn luyện các mô hình phân loại, tính Độ mới lạ ($IU, KL, BD$), cảm xúc và nhận diện bối cảnh đính chính. | Trọng số các mô hình đã huấn luyện và module xử lý độ bất định. |
| 6 | Tuần 10 | Cài đặt Giai đoạn 4 (`evidence_analyzer.py`): Suy luận 3 nhãn `Supported`/`Refuted`/`NEI`, đối sánh `NLI Verifier` với `LLM Few-shot Evidence-Grounded Verification`. | Engine kiểm chứng bằng chứng, giải thích và trích dẫn nguồn. |
| 7 | Tuần 11 – Tuần 12 | Cài đặt Giai đoạn 5 (`main.py`, `App.tsx`, `TrainingPage.tsx`): Xây dựng API, giao diện Web hiển thị Confidence/Citation/Date và tính năng lưu **User Feedback (`SQLite`)**. | Hệ thống Web Prototype hoàn chỉnh. |
| 8 | Tuần 13 | Chạy bộ Benchmark thực nghiệm (`evaluate_benchmark.py`) đo đủ 7 chỉ số: `Claim Detection F1`, `Recall@K/MRR/NDCG`, `Verification F1`, `Evidence P/R`, `Citation Accuracy`, `E2E Accuracy`, `Latency`. | Bảng số liệu và biểu đồ thực nghiệm toàn diện. |
| 9 | Tuần 14 | Hoàn thiện quyển báo cáo Đồ án tốt nghiệp, rà soát mã nguồn và chuẩn bị slide bảo vệ trước Hội đồng. | Quyển báo cáo Đồ án tốt nghiệp và bộ sản phẩm hoàn chỉnh. |

*Hà Nội, ngày …. tháng …. năm 2026*

| CÁN BỘ HƯỚNG DẪN | SINH VIÊN THỰC HIỆN |
| :---: | :---: |
| **ThS. Thái Thị Thanh Vân** | **Bùi Đức Anh** |
