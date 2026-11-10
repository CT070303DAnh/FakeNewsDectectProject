# BAN CƠ YẾU CHÍNH PHỦ
# HỌC VIỆN KỸ THUẬT MẬT MÃ
¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯

## ĐỀ CƯƠNG ĐỒ ÁN TỐT NGHIỆP
**Đề tài: Xây dựng hệ thống AI hỗ trợ kiểm chứng thông tin tiếng Việt dựa trên truy xuất và phân tích bằng chứng**

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

Các phương pháp phát hiện tin giả truyền thống thường tiếp cận dưới góc độ phân loại văn bản thuần túy (Text Classification) dựa trên đặc trưng từ vựng hoặc văn phong. Tuy nhiên, hướng tiếp cận này bộc lộ hạn chế lớn trong thực tiễn: các mô hình học máy dễ bị đánh lừa bởi những tin giả được ngụy trang dưới văn phong báo chí trang trọng, hoặc ngược lại, nhận diện nhầm (False Positive) các bài báo chính thống đang trích dẫn lại tin đồn để đính chính/bác bỏ là tin giả. Đặc biệt, đối với các sự kiện mới phát sinh hoặc tin giả đánh tráo mốc thời gian, một mô hình phân loại tĩnh không thể xác định tính đúng/sai của sự kiện nếu không đối chiếu với nguồn bằng chứng xác thực bên ngoài.

Xuất phát từ thực tế đó, xu hướng nghiên cứu hiện đại chuyển dịch sang **kiểm chứng thông tin dựa trên truy xuất và phân tích bằng chứng (Evidence-Grounded Fact Verification)**. Thay vì chỉ dự đoán cảm tính qua mặt chữ, hệ thống thực hiện phát hiện và phân loại các phát biểu đáng kiểm chứng (*Claim Detection / Check-worthiness*), chuẩn hóa mệnh đề (*Claim Normalization*), truy xuất bằng chứng lai (*Hybrid Retrieval: BM25 + Dense Embedding*) kết hợp xếp hạng lại (*Cross-Encoder Reranking*) và lọc theo mốc thời gian (*Temporal Filtering*), sau đó áp dụng suy luận ngôn ngữ tự nhiên (*Natural Language Inference – NLI*) và đối sánh với *LLM Few-Shot Evidence-Grounded Verification* để đánh giá mệnh đề theo 3 mức chuẩn: `Supported` (Được chứng minh), `Refuted` (Bác bỏ) và `Not Enough Evidence – NEI` (Chưa đủ bằng chứng).

Đề tài **"Xây dựng hệ thống AI hỗ trợ kiểm chứng thông tin tiếng Việt dựa trên truy xuất và phân tích bằng chứng"** giải quyết trọn vẹn bài toán xác thực thông tin có căn cứ minh bạch, đồng thời kết hợp các chỉ số đo độ mới lạ thông tin (*Information Uniqueness, KL-Divergence, Bhattacharyya Distance*), phân tích cảm xúc/phản bác theo phương pháp luận của *Vosoughi et al. (2018)* và cơ chế thu thập phản hồi người dùng (*Human-in-the-loop User Feedback*). Qua đó, hệ thống mang lại một giải pháp kiểm chứng toàn diện, xử lý tốt độ bất định (*Uncertainty Handling*), cung cấp trích dẫn nguồn (*Citation*) kèm thời điểm bằng chứng (*Evidence Date*) để người dùng tự đánh giá.

### 2. Mục tiêu nghiên cứu của đề tài
- Xây dựng hệ thống AI có khả năng tự động xác định các phát biểu có thể kiểm chứng (*Check-worthy Claims*), chuẩn hóa mệnh đề, truy xuất và xếp hạng bằng chứng từ kho dữ liệu được kiểm soát (Controlled Evidence Corpus) kết hợp báo chí chính thống, và đánh giá phát biểu theo 3 mức `Supported` / `Refuted` / `Not Enough Evidence (NEI)`.
- Thiết kế và triển khai kiến trúc 5 giai đoạn (12 Node xử lý) tích hợp: Phân loại *Check-worthiness*, Truy xuất lai (*BM25 Baseline + Dense Semantic Retrieval + Cross-Encoder Reranker + Temporal Filtering*), Suy luận *NLI 3-nhãn* đối sánh với *LLM Few-Shot Evidence-Grounded Verification*, Xử lý độ bất định (*Uncertainty Handling* qua Shannon Entropy & Độ mới lạ $IU, KL, BD$) và Phản hồi người dùng (*User Feedback*).
- Xây dựng bộ dữ liệu đánh giá kiểm chứng thông tin tiếng Việt (**Vietnamese Fact Verification Evaluation Set – VN-FactCheck-350** gồm 350 claims chất lượng cao và gần 500 tài liệu bằng chứng chuẩn) với đầy đủ các trường: `claim`, `label`, `gold_evidence`, `source`, `date`, `topic`.
- Thực nghiệm và đánh giá toàn diện hệ thống trên bộ tiêu chí chuẩn quốc tế: `Claim Detection Macro-F1`, `Retrieval Recall@K / MRR / NDCG`, `Verification Macro-F1`, `Evidence Precision/Recall`, `Citation Accuracy`, `End-to-End Evidence-Grounded Accuracy` và `Latency (ms)`.

### 3. Đối tượng và phạm vi nghiên cứu
- **Đối tượng nghiên cứu:**
  + Bài toán phát hiện và phân loại câu đáng kiểm chứng (*Claim Detection & Check-worthiness Text Classification*), trích xuất và chuẩn hóa mệnh đề (*Claim Extraction & Normalization*).
  + Các kỹ thuật truy xuất và xếp hạng bằng chứng: *Okapi BM25 Baseline*, *Dense Semantic Retrieval* (`Sentence-BERT`/`PhoBERT`/`SVD-128`), *Hybrid Retrieval* và *Cross-Encoder Pairwise Reranker* kèm kiểm tra khớp mốc thời gian (*Temporal Filtering*).
  + Bài toán kiểm chứng dựa trên bằng chứng (*Evidence-Grounded Verification*): Phân loại 3 nhãn `Supported` / `Refuted` / `NEI` bằng mô hình NLI/Transformer kết hợp đối sánh với *LLM Few-Shot Grounded Prompting*, xử lý độ bất định (*Uncertainty Handling*) và kiểm tra độ chính xác trích dẫn (*Citation Accuracy*).
  + Các chỉ số độ mới lạ thông tin (*Information Uniqueness – IU, Symmetric KL-Divergence – KL, Bhattacharyya Distance – BD*), phân tích cảm xúc/phản bác (*Vosoughi et al., Science 2018*) và dàn 8 mô hình Học máy / Học sâu phân tích văn phong tiếng Việt (`Logistic Regression`, `SVM`, `Random Forest`, `Naive Bayes`, `Decision Tree`, `PhoBERT`, `ViBERT`, `Vietnamese-SBERT`).
- **Phạm vi nghiên cứu:**
  + Kiểm chứng các bài viết, phát biểu hoặc URL tin tức tiếng Việt thuộc các chủ đề trọng tâm: Chính sách – Pháp luật, Tài chính – Ngân hàng, Y tế – Sức khỏe, Giáo dục – Tuyển sinh, Giao thông – Đô thị và An ninh mạng – Cảnh báo lừa đảo.
  + Truy xuất bằng chứng trên Kho bằng chứng chuẩn được kiểm soát (`verified_facts_vn.json` & `VN-FactCheck-350 Corpus` gồm ~500 hồ sơ có metadata `source`, `url`, `date`, `topic`) kết hợp nguồn báo chí chính thống Việt Nam trực tuyến.
  + Phát triển sản phẩm hoàn chỉnh gồm: Web Prototype (`React`/`TypeScript`/`TailwindCSS`), RESTful API & SSE Training Server (`FastAPI`), Kho bằng chứng & Bộ dữ liệu đánh giá 350 claims, cùng bộ kịch bản kiểm thử và báo cáo thực nghiệm.

### 4. Các nhiệm vụ chính cần thực hiện
Nội dung nghiên cứu của đồ án tập trung vào các nhiệm vụ chính sau:
- Nghiên cứu cơ sở lý thuyết về kiểm chứng thông tin tự động dựa trên bằng chứng (*Automated Evidence-Based Fact-Checking*), xử lý ngôn ngữ tự nhiên tiếng Việt (`underthesea`, chuẩn hóa Unicode NFC/Teencode, NER) và phương pháp luận từ bài báo *Vosoughi et al. (Science, 2018)*.
- Xây dựng bộ dữ liệu **Vietnamese Fact Verification Evaluation Set (`VN-FactCheck-350`)** gồm 350 mệnh đề kiểm chứng chất lượng cao gắn liền với bằng chứng chuẩn (`claim`, `label` thuộc 3 nhãn `Supported`/`Refuted`/`NEI`, `gold_evidence`, `source`, `date`, `topic`).
- Thiết kế và cài đặt hệ thống theo kiến trúc 5 giai đoạn (12 Node xử lý) giữ nguyên tính mô-đun hóa:
  1. *Giai đoạn 1 (Nodes 1–3):* Tiếp nhận bài viết/URL, chuẩn hóa tiếng Việt, phát hiện câu đáng kiểm chứng (*Check-worthiness Classification*) và chuẩn hóa claim (*Claim Normalization*).
  2. *Giai đoạn 2 (Nodes 4–5):* Truy xuất và xếp hạng bằng chứng với 4 chế độ đối chứng (*BM25 Baseline*, *Dense Retrieval*, *Hybrid Retrieval*, *Hybrid + Cross-Encoder Reranker*) kết hợp lọc theo thời điểm bằng chứng (*Temporal Filtering*).
  3. *Giai đoạn 3 (Nodes 6–8):* Phân tích tín hiệu nội tại, đo Độ mới lạ thông tin ($IU, KL, BD$), cảm xúc/phản bác, dự đoán văn phong bằng 8 mô hình ML/DL và nhận diện ngữ cảnh bài báo đính chính (*Debunking Context*).
  4. *Giai đoạn 4 (Nodes 9–11):* Suy luận quan hệ NLI 3 nhãn (`Supported` / `Refuted` / `NEI`), đối sánh với *LLM Few-Shot Evidence-Grounded Verification*, tính toán độ bất định Shannon Entropy (*Uncertainty Handling*) và tổng hợp phán quyết lai.
  5. *Giai đoạn 5 (Node 12):* Hiển thị độ tin cậy (`confidence`), giải thích từng bước (*Grounded Explanation*), trích dẫn nguồn kèm ngày công bố (*Citation & Evidence Date*), đo độ trễ (`Latency`) và thu thập phản hồi người dùng (*User Feedback*).
- Thực nghiệm và đánh giá định lượng toàn bộ các thành phần theo đúng 7 nhóm chỉ số chuẩn: `Claim Detection Macro-F1`, `Retrieval Recall@K / MRR / NDCG`, `Verification Macro-F1`, `Evidence Precision/Recall`, `Citation Accuracy`, `End-to-End Evidence-Grounded Accuracy` và `Latency`.

### 5. Kết quả dự kiến (Sản phẩm đầu ra)
- **Web Prototype & API Server:** Ứng dụng Web (`React/TypeScript`) và hệ thống API (`FastAPI` trên Port 8000 & 8001) hỗ trợ đầy đủ 10 chức năng: nhập bài viết/claim, phát hiện & chuẩn hóa claim, truy xuất & xếp hạng bằng chứng, phân loại `Supported`/`Refuted`/`NEI`, hiển thị `confidence` & độ bất định, giải thích có căn cứ, trích dẫn nguồn (`citation`), hiển thị thời điểm bằng chứng (`date`) và lưu phản hồi người dùng (`user feedback`).
- **Evidence Corpus & Evaluation Dataset (`VN-FactCheck-350`):** Kho bằng chứng kiểm soát (~500 tài liệu) và tập dữ liệu đánh giá gồm 350 claims tiếng Việt có đủ 6 trường (`claim`, `label`, `gold_evidence`, `source`, `date`, `topic`).
- **Retrieval & Verifier Models:** Bộ mô-đun truy xuất (`BM25`, `Dense SVD-128/SBERT`, `Hybrid + Reranker`), bộ phân loại `Check-worthiness`, mô hình kiểm chứng `NLI 3-nhãn` so sánh trực tiếp với `LLM Few-Shot Grounded Verifier` và 8 mô hình phân loại văn phong.
- **Bộ Test & Báo cáo thực nghiệm:** Kịch bản benchmark tự động (`evaluate_benchmark.py` & `test_pipeline.py`) xuất báo cáo đầy đủ các chỉ số `Macro-F1`, `Recall@K`, `MRR`, `NDCG`, `Citation Accuracy`, `End-to-End Grounded Accuracy` và `Latency`.

---

## II. DỰ KIẾN CÁC CHƯƠNG MỤC

**MỤC LỤC**
**DANH MỤC CÁC TỪ VIẾT TẮT**
**DANH MỤC CÁC BẢNG BIỂU**
**DANH MỤC CÁC HÌNH VẼ**
**LỜI CẢM ƠN**
**MỞ ĐẦU**

**CHƯƠNG 1. TỔNG QUAN ĐỀ TÀI VÀ CƠ SỞ LÝ THUYẾT**
* 1.1. Tổng quan về bài toán kiểm chứng thông tin dựa trên bằng chứng (Evidence-Based Fact Verification)
  * 1.1.1. Sự dịch chuyển từ phát hiện tin giả dựa trên văn phong sang kiểm chứng dựa trên bằng chứng
  * 1.1.2. Phương pháp luận xác thực thông tin và độ mới lạ từ nghiên cứu của Vosoughi et al. (Science, 2018)
  * 1.1.3. Quy trình chuẩn của hệ thống Fact-Checking: Claim Detection $\rightarrow$ Evidence Retrieval $\rightarrow$ NLI Verification $\rightarrow$ Grounded Explanation
* 1.2. Cơ sở lý thuyết về Truy xuất thông tin (IR) và Suy luận ngôn ngữ tự nhiên (NLI) cho tiếng Việt
  * 1.2.1. Bài toán phát hiện phát biểu đáng kiểm chứng (Check-worthiness Classification) và chuẩn hóa mệnh đề tiếng Việt
  * 1.2.2. Truy xuất từ khóa (Okapi BM25), Truy xuất ngữ nghĩa đặc (Dense Retrieval) và Truy xuất lai (Hybrid Retrieval + Cross-Encoder Reranking)
  * 1.2.3. Lọc bằng chứng theo ngữ cảnh thời gian (Metadata & Temporal Filtering)
  * 1.2.4. Mô hình suy luận 3 nhãn (`Supported` / `Refuted` / `NEI`), Xử lý độ bất định (Uncertainty Entropy) và Đối sánh với LLM Few-Shot Grounded Verification
  * 1.2.5. Các độ đo Độ mới lạ thông tin ($IU, KL, BD$), Cảm xúc và 8 mô hình Học máy / Học sâu phân tích văn phong
* 1.3. Các công cụ, framework và tiêu chí đánh giá hệ thống (`Recall@K`, `MRR`, `NDCG`, `Macro-F1`, `FEVER Grounded Score`)
* 1.4. Tổng kết chương 1

**CHƯƠNG 2. PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG**
* 2.1. Phát biểu bài toán và đặc tả nghiệp vụ kiểm chứng thông tin tiếng Việt
* 2.2. Phân tích yêu cầu chức năng (10 chức năng cốt lõi) và yêu cầu phi chức năng
* 2.3. Thiết kế kiến trúc tổng thể hệ thống (Pipeline 5 Giai đoạn – 12 Node xử lý)
* 2.4. Thiết kế biểu đồ Use Case, biểu đồ Hoạt động và biểu đồ Tuần tự (Sequence Diagrams)
* 2.5. Thiết kế cấu trúc Kho bằng chứng kiểm soát (Evidence Corpus) và Bộ dữ liệu đánh giá `VN-FactCheck-350` (`claim`, `label`, `gold_evidence`, `source`, `date`, `topic`)
* 2.6. Thiết kế thuật toán Hybrid Retrieval + Reranker, NLI Verifier và vòng lặp Phản hồi người dùng (Human-in-the-loop Feedback)
* 2.7. Tổng kết chương 2

**CHƯƠNG 3. XÂY DỰNG VÀ CÀI ĐẶT HỆ THỐNG**
* 3.1. Xây dựng Giai đoạn 1 (`preprocess.py`): Phát hiện câu đáng kiểm chứng (Check-worthiness), chuẩn hóa claim và trích xuất thực thể/thời gian
* 3.2. Xây dựng Giai đoạn 2 (`evidence_retriever.py`): Cài đặt BM25 Baseline, Dense Retrieval, Hybrid Retrieval + Cross-Encoder Reranker và Temporal Filtering
* 3.3. Xây dựng Giai đoạn 3 (`novelty_emotion.py`, `train.py`): Đo Độ mới lạ ($IU, KL, BD$), phân tích cảm xúc/ngữ cảnh đính chính và huấn luyện các mô hình phân loại
* 3.4. Xây dựng Giai đoạn 4 (`evidence_analyzer.py`): Kiểm chứng NLI 3 nhãn (`Supported`/`Refuted`/`NEI`), đối sánh LLM Few-Shot Grounded, tính độ bất định và kiểm tra Citation
* 3.5. Xây dựng Giai đoạn 5 (`main.py`, `App.tsx`, `TrainingPage.tsx`): Giao diện giải trình bằng chứng, hiển thị ngày công bố (`date`), độ trễ (`latency`) và thu thập `User Feedback`
* 3.6. Tổng kết chương 3

**CHƯƠNG 4. THỰC NGHIỆM, TRIỂN KHAI VÀ ĐÁNH GIÁ KẾT QUẢ**
* 4.1. Xây dựng và thống kê bộ dữ liệu đánh giá `VN-FactCheck-350` (350 claims & 499 tài liệu bằng chứng)
* 4.2. Thực nghiệm đánh giá Giai đoạn 1: Kết quả phân loại Check-worthiness (`Claim Detection Macro-F1`)
* 4.3. Thực nghiệm đánh giá Giai đoạn 2: So sánh `BM25 Baseline` vs. `Dense` vs. `Hybrid + Reranker` qua `Recall@1`, `Recall@3`, `Recall@5`, `MRR@5` và `NDCG@5`
* 4.4. Thực nghiệm đánh giá Giai đoạn 4 & Toàn trình: So sánh `NLI Verifier` vs. `LLM Few-Shot Grounded` vs. `VietFactCheck Hybrid` qua `Verification Macro-F1`, `Evidence Precision/Recall`, `Citation Accuracy`, `End-to-End Grounded Accuracy` và `Latency`
* 4.5. Tổng kết chương 4

**KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**
**TÀI LIỆU THAM KHẢO**
**PHỤ LỤC**

---

## III. TÀI LIỆU THAM KHẢO ĐỂ XÂY DỰNG ĐỀ CƯƠNG

[1] Vosoughi, S., Roy, D., & Aral, S. (2018). *The spread of true and false news online*. Science, 359(6380), 1146–1151. DOI: 10.1126/science.aap9559.
[2] Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification*. In Proceedings of NAACL-HLT 2018, pp. 809–819.
[3] Guo, Z., Schlichtkrull, M., & Vlachos, A. (2022). *A Survey on Automated Fact-Checking*. Transactions of the Association for Computational Linguistics (TACL), 10, 178–206.
[4] Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389.
[5] Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese*. In Findings of ACL: EMNLP 2020 (pp. 1037–1042).
[6] Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. In Proceedings of EMNLP-IJCNLP 2019, pp. 3982–3992.
[7] Le, D. T., Vu, X. S., & Nguyen, M. L. (2020). *ReINTEL: A Multimodal Data Challenge for Responsible Information Identification on Vietnamese Social Media*. In Proceedings of VLSP 2020.

---

## IV. KẾ HOẠCH THỰC HIỆN

| STT | Thời gian | Nội dung thực hiện | Kết quả dự kiến |
| :---: | :---: | :--- | :--- |
| 1 | Tuần 1 – Tuần 2 | Nghiên cứu bài báo *Vosoughi et al. (Science, 2018)*, quy trình kiểm chứng thông tin dựa trên bằng chứng (FEVER, Hybrid IR, NLI) và các mô hình NLP tiếng Việt. | Tài liệu tổng hợp cơ sở lý thuyết và đề cương chi tiết. |
| 2 | Tuần 3 – Tuần 4 | Xây dựng Kho bằng chứng chuẩn (`verified_facts_vn.json`) và Bộ dữ liệu đánh giá **Vietnamese Fact Verification Evaluation Set (`VN-FactCheck-350`)** gồm 350 claims có đủ `claim`, `label`, `gold_evidence`, `source`, `date`, `topic`. | Bộ dữ liệu đánh giá 350 claims và Corpus ~500 văn bản bằng chứng. |
| 3 | Tuần 5 | Thiết kế kiến trúc chi tiết 5 Giai đoạn (12 Node xử lý) và xây dựng Giai đoạn 1 (`preprocess.py`): Phát hiện câu đáng kiểm chứng (*Check-worthiness*), chuẩn hóa claim và trích xuất thực thể/thời gian. | Module Claim Detection & Normalization đạt `Macro-F1 > 85%`. |
| 4 | Tuần 6 – Tuần 7 | Xây dựng Giai đoạn 2 (`evidence_retriever.py`): Cài đặt `BM25 Baseline`, `Dense Retrieval`, `Hybrid Retrieval`, `Cross-Encoder Reranker` và bộ lọc thời gian (`Temporal Filtering`). | Module truy xuất đa chế độ và kết quả đo `Recall@K`, `MRR`, `NDCG`. |
| 5 | Tuần 8 – Tuần 9 | Xây dựng Giai đoạn 3 (`novelty_emotion.py`, `train.py`): Tính toán Độ mới lạ thông tin ($IU, KL, BD$), phân tích cảm xúc/phản bác, nhận diện ngữ cảnh đính chính và huấn luyện dàn mô hình phân loại. | Bộ trọng số mô hình ML/DL và module Novelty/Debunking. |
| 6 | Tuần 10 | Xây dựng Giai đoạn 4 (`evidence_analyzer.py`): Phân loại NLI 3 nhãn (`Supported`/`Refuted`/`NEI`), đối sánh với `LLM Few-Shot Evidence-Grounded Verification`, xử lý độ bất định (`Uncertainty`) và kiểm tra `Citation`. | Engine suy luận bằng chứng và đối sánh NLI vs. LLM Few-Shot. |
| 7 | Tuần 11 – Tuần 12 | Xây dựng Giai đoạn 5 (`main.py`, `training_api.py`, `App.tsx`, `TrainingPage.tsx`): Giao diện hiển thị `confidence`, giải thích, trích dẫn kèm `date`, đo `latency` và chức năng gửi `User Feedback`. | Hệ thống Web Prototype & API hoàn chỉnh. |
| 8 | Tuần 13 | Chạy kịch bản thực nghiệm toàn diện (`evaluate_benchmark.py`), tổng hợp bảng số liệu 7 nhóm tiêu chí đánh giá và phân tích kết quả thực nghiệm. | Báo cáo thực nghiệm đầy đủ các chỉ số IR, NLI và End-to-End. |
| 9 | Tuần 14 | Hoàn thiện quyển báo cáo Đồ án tốt nghiệp, rà soát mã nguồn và chuẩn bị slide bảo vệ trước Hội đồng. | Quyển báo cáo Đồ án tốt nghiệp và bộ source code hoàn chỉnh. |

*Hà Nội, ngày …. tháng …. năm 2026*

| CÁN BỘ HƯỚNG DẪN | SINH VIÊN THỰC HIỆN |
| :---: | :---: |
| **ThS. Thái Thị Thanh Vân** | **Bùi Đức Anh** |
