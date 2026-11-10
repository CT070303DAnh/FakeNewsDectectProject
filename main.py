from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import joblib
import os
import sys
import json
import time
from datetime import datetime
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    HAS_TORCH = False

try:
    from gensim.models.doc2vec import Doc2Vec
    HAS_GENSIM = True
except ImportError:
    Doc2Vec = None
    HAS_GENSIM = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.preprocess import (
    clean_text,
    extract_text_from_url_if_needed,
    extract_claims_and_queries,
)
from src.evidence_retriever import MultiTierEvidenceRetriever
from src.novelty_emotion import compute_novelty_metrics, analyze_emotion_and_heuristics
from src.evidence_analyzer import synthesize_hybrid_verdict

app = FastAPI(
    title="Hệ thống AI Hỗ trợ Kiểm chứng Thông tin Tiếng Việt Dựa trên Truy xuất & Phân tích Bằng chứng"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
models = {}
phobert_model = None
phobert_tokenizer = None
vibert_model = None
vibert_tokenizer = None
sbert_model = None
sbert_tokenizer = None
metrics = {}
d2v_model = None
background_d2v_centroid = None
evidence_retriever: MultiTierEvidenceRetriever = None


def load_resources():
    global models, metrics, d2v_model, background_d2v_centroid
    global phobert_model, phobert_tokenizer, vibert_model, vibert_tokenizer, sbert_model, sbert_tokenizer
    global evidence_retriever

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"DEBUG: Nạp tài nguyên từ: {MODELS_DIR}")

    # Khởi tạo Module Truy xuất Bằng chứng Đa tầng (Nodes 4, 5, 6)
    if evidence_retriever is None:
        evidence_retriever = MultiTierEvidenceRetriever(BASE_DIR)

    try:
        metrics_path = os.path.join(MODELS_DIR, "models_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        model_names = ["logistic_regression", "svm", "decision_tree", "naive_bayes", "random_forest"]
        for name in model_names:
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            if os.path.exists(path):
                try:
                    models[name] = joblib.load(path)
                except Exception as model_e:
                    print(f"DEBUG: Lỗi nạp {name}: {model_e}")

        # Load PhoBERT, ViBERT, SBERT (nếu môi trường có cài đặt PyTorch & Transformers)
        if HAS_TORCH:
            phobert_path = os.path.join(MODELS_DIR, "phobert_model")
            if os.path.exists(phobert_path):
                try:
                    phobert_tokenizer = AutoTokenizer.from_pretrained(phobert_path)
                    phobert_model = AutoModelForSequenceClassification.from_pretrained(phobert_path)
                    phobert_model.eval()
                except Exception as pb_e:
                    print(f"DEBUG: Lỗi nạp PhoBERT: {pb_e}")

            vibert_path = os.path.join(MODELS_DIR, "vibert_model")
            if os.path.exists(vibert_path):
                try:
                    vibert_tokenizer = AutoTokenizer.from_pretrained(vibert_path)
                    vibert_model = AutoModelForSequenceClassification.from_pretrained(vibert_path)
                    vibert_model.eval()
                except Exception as vb_e:
                    print(f"DEBUG: Lỗi nạp ViBERT: {vb_e}")

            sbert_path = os.path.join(MODELS_DIR, "sbert_model")
            if os.path.exists(sbert_path):
                try:
                    sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_path)
                    sbert_model = AutoModelForSequenceClassification.from_pretrained(sbert_path)
                    sbert_model.eval()
                except Exception as sb_e:
                    print(f"DEBUG: Lỗi nạp SBERT: {sb_e}")

        # Load Doc2Vec & tính Vector Nền Γ_b cho Module Novelty (Mục S7.2)
        d2v_path = os.path.join(MODELS_DIR, "doc2vec.model")
        if HAS_GENSIM and os.path.exists(d2v_path):
            d2v_model = Doc2Vec.load(d2v_path)
            if hasattr(d2v_model, "dv") and len(d2v_model.dv) > 0:
                sample_count = min(500, len(d2v_model.dv))
                vecs = [d2v_model.dv[i] for i in range(sample_count)]
                background_d2v_centroid = np.mean(vecs, axis=0)

    except Exception as e:
        print(f"DEBUG: Lỗi tổng quát khi nạp tài nguyên: {e}")


load_resources()


class NewsItem(BaseModel):
    text: str


class FeedbackItem(BaseModel):
    claim: str
    predicted_verdict: str
    predicted_nli_label: Optional[str] = "NEI"
    user_agreement: bool
    user_suggested_label: Optional[str] = None
    user_comment: Optional[str] = ""
    evidence_source_url: Optional[str] = ""


EDUCATIONAL_CONTENT = {
    "logistic_regression": {
        "name": "Logistic Regression (Hồi quy Logistic)",
        "concept": "Hồi quy Logistic là thuật toán phân loại thống kê dự đoán xác suất văn bản thuộc lớp Tin giả hoặc Tin thật.",
        "principle": "Sử dụng hàm Sigmoid trên tổ hợp tuyến tính trọng số TF-IDF để ánh xạ đầu ra vào khoảng [0, 1].",
        "flow": "Văn bản -> TF-IDF -> Tổng trọng số -> Hàm Sigmoid -> Xác suất -> Phân loại."
    },
    "svm": {
        "name": "Support Vector Machine (Máy vector hỗ trợ)",
        "concept": "SVM tìm kiếm siêu phẳng tối ưu trong không gian đặc trưng TF-IDF nhiều chiều.",
        "principle": "Tối đa hóa khoảng cách lề (Margin) giữa các điểm dữ liệu biên (Support Vectors) của lớp Tin thật và Tin giả.",
        "flow": "Văn bản -> TF-IDF -> Ánh xạ không gian -> Tìm siêu phẳng -> Phân loại."
    },
    "decision_tree": {
        "name": "Decision Tree (Cây quyết định)",
        "concept": "Cây quyết định phân tách dữ liệu đệ quy dựa trên độ lợi thông tin (Information Gain / Gini).",
        "principle": "Kiểm tra sự xuất hiện của các cụm từ khóa đặc trưng tại từng nút để đi đến kết luận ở nút lá.",
        "flow": "Văn bản -> TF-IDF -> Kiểm tra nút -> Đi theo nhánh -> Kết luận tại lá."
    },
    "naive_bayes": {
        "name": "Naive Bayes (Định lý Bayes)",
        "concept": "Mô hình xác suất dựa trên định lý Bayes với giả định độc lập có điều kiện giữa các từ vựng.",
        "principle": "Tính xác suất hậu nghiệm của lớp Tin giả dựa trên tần suất đồng xuất hiện của từ ngữ.",
        "flow": "Văn bản -> TF-IDF -> Tính xác suất Bayes -> So sánh -> Chọn lớp."
    },
    "random_forest": {
        "name": "Random Forest (Rừng ngẫu nhiên)",
        "concept": "Tập hợp (Ensemble) hàng trăm cây quyết định huấn luyện theo kỹ thuật Bagging.",
        "principle": "Giảm phương sai và chống quá khớp (Overfitting) bằng cách lấy phiếu bầu đa số từ nhiều cây độc lập.",
        "flow": "Văn bản -> TF-IDF -> Qua hàng trăm cây -> Voting -> Kết luận."
    },
    "phobert": {
        "name": "PhoBERT (Transformer cho tiếng Việt - VinAI)",
        "concept": "Mô hình ngôn ngữ tiền huấn luyện RoBERTa chuyên biệt cho tiếng Việt đã tách từ ghép.",
        "principle": "Sử dụng cơ chế Self-Attention đa đầu để nắm bắt ngữ nghĩa phụ thuộc ngữ cảnh hai chiều.",
        "flow": "Văn bản -> Tách từ -> Token hóa -> Transformer Layers -> Attention -> Phân loại."
    },
    "vibert": {
        "name": "ViBERT (BERT cho tiếng Việt - FPTAI)",
        "concept": "Biến thể BERT-base được huấn luyện trên kho dữ liệu báo chí tiếng Việt lớn.",
        "principle": "Học biểu diễn ngữ cảnh qua tác vụ Masked Language Modeling và tinh chỉnh cho bài toán phân loại.",
        "flow": "Văn bản -> BERT Tokenizer -> Encoder Layers -> Contextual Vector -> Phân loại."
    },
    "sbert": {
        "name": "Vietnamese-SBERT (Sentence-BERT)",
        "concept": "Mô hình mạng Siamese/Triplet tạo vector biểu diễn ngữ nghĩa cấp độ câu cho tiếng Việt.",
        "principle": "Tối ưu hóa khoảng cách Cosine giữa các mệnh đề để phục vụ cả phân loại lẫn truy xuất bằng chứng.",
        "flow": "Văn bản -> Tokenizer -> Transformer Backbone -> Mean Pooling -> Sentence Vector."
    },
    "doc2vec": {
        "name": "Doc2Vec (Paragraph Vector - Mục S2.2 & S7.2)",
        "concept": "Thuật toán biểu diễn đoạn văn thành vector đặc (Dense Vector) dùng trong bài báo Science (2018).",
        "principle": "Kết hợp học ngữ cảnh từ vựng và ID đoạn văn để đo khoảng cách thông tin (IU, KL, BD).",
        "flow": "Văn bản -> Tokenize -> Mạng Nơ-ron -> Vector 100 chiều -> Tính Novelty (IU, KL, BD)."
    }
}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "evidence_corpus_size": len(evidence_retriever.local_corpus) if evidence_retriever else 0,
        "deep_learning_loaded": {
            "phobert": phobert_model is not None,
            "vibert": vibert_model is not None,
            "sbert": sbert_model is not None,
        },
    }


@app.post("/predict")
def predict(item: NewsItem):
    t_start = time.perf_counter()
    if not models and phobert_model is None and vibert_model is None and sbert_model is None:
        load_resources()

    # =================================================================
    # GIAI ĐOẠN 1 (Nodes 1, 2, 3): Tiếp nhận, Chuẩn hóa & Tách Mệnh đề
    # =================================================================
    url_info = extract_text_from_url_if_needed(item.text)
    effective_raw_text = url_info["text"]
    cleaned_text = clean_text(effective_raw_text)
    claim_analysis = extract_claims_and_queries(effective_raw_text)
    claim_analysis["is_url_input"] = url_info["is_url"]
    claim_analysis["source_url"] = url_info["source_url"]

    # =================================================================
    # GIAI ĐOẠN 2 (Nodes 4, 5, 6): Truy xuất Bằng chứng Đa tầng & Lọc Ngưỡng S2.2
    # =================================================================
    retrieval_output = evidence_retriever.retrieve_and_filter_by_vosoughi_threshold(
        claim_text=claim_analysis["primary_claim"],
        queries=claim_analysis["search_queries"],
        claim_year=claim_analysis.get("claim_temporal_year"),
    )

    # =================================================================
    # GIAI ĐOẠN 3 (Nodes 7, 8, 9): Dự đoán Đa mô hình, Đo Độ mới lạ & Cảm xúc
    # =================================================================
    results = {}

    # Node 7A: Dự đoán với 5 mô hình Scikit-learn
    for name in ["logistic_regression", "svm", "decision_tree", "naive_bayes", "random_forest"]:
        if name in models:
            try:
                pipeline = models[name]
                probs = pipeline.predict_proba([cleaned_text])[0]
                label = int(pipeline.predict([cleaned_text])[0])

                raw_metrics = metrics.get(name, {})
                model_metrics = {
                    "accuracy": float(raw_metrics.get("accuracy", 0)),
                    "f1": float(raw_metrics.get("f1", 0)),
                    "recall": float(raw_metrics.get("recall", 0)),
                    "precision": float(raw_metrics.get("precision", 0)),
                    "false_alarm_rate": float(raw_metrics.get("false_alarm_rate", 0)),
                    "confusion_matrix": raw_metrics.get("confusion_matrix", [[0, 0], [0, 0]]),
                }

                results[name] = {
                    "prediction": "Fake" if label == 1 else "Real",
                    "fake_probability": float(probs[1]),
                    "real_probability": float(probs[0]),
                    "metrics": model_metrics,
                    "education": EDUCATIONAL_CONTENT.get(name, {}),
                }
            except Exception as e:
                print(f"DEBUG: Lỗi dự đoán {name}: {e}")

    # Node 7B: Dự đoán với 3 mô hình Deep Learning (PhoBERT, ViBERT, SBERT)
    dl_models = [
        ("phobert", phobert_model, phobert_tokenizer),
        ("vibert", vibert_model, vibert_tokenizer),
        ("sbert", sbert_model, sbert_tokenizer),
    ]
    for name, model, tokenizer in dl_models:
        if model and tokenizer:
            try:
                inputs = tokenizer(
                    cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                    label = int(np.argmax(probs))

                raw_metrics = metrics.get(name, {})
                model_metrics = {
                    "accuracy": float(raw_metrics.get("accuracy", 0)),
                    "f1": float(raw_metrics.get("f1", 0)),
                    "recall": float(raw_metrics.get("recall", 0)),
                    "precision": float(raw_metrics.get("precision", 0)),
                    "false_alarm_rate": float(raw_metrics.get("false_alarm_rate", 0)),
                    "confusion_matrix": raw_metrics.get("confusion_matrix", [[0, 0], [0, 0]]),
                }

                results[name] = {
                    "prediction": "Fake" if label == 1 else "Real",
                    "fake_probability": float(probs[1]),
                    "real_probability": float(probs[0]),
                    "metrics": model_metrics,
                    "education": EDUCATIONAL_CONTENT.get(name, {}),
                }
            except Exception as e:
                print(f"DEBUG: Lỗi dự đoán DL {name}: {e}")

    # Node 8: Tính Độ mới lạ thông tin (Novelty: IU, KL, BD - Mục S7.2 Science 2018)
    d2v_vector = []
    if d2v_model:
        inferred_vec = d2v_model.infer_vector(cleaned_text.split())
        d2v_vector = inferred_vec.tolist()
        novelty_metrics = compute_novelty_metrics(inferred_vec, background_d2v_centroid)
    else:
        # Tính phân phối chủ đề dựa trên độ tương đồng với 100 mẫu tin nền đầu tiên trong kho
        if evidence_retriever and evidence_retriever.vectorizer is not None and evidence_retriever.corpus_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_tfidf = evidence_retriever.vectorizer.transform([cleaned_text])
            n_topics = min(100, evidence_retriever.corpus_matrix.shape[0])
            ref_mat = evidence_retriever.corpus_matrix[:n_topics]
            claim_topic_dist = cosine_similarity(q_tfidf, ref_mat)[0] + 0.05
            centroid_tfidf = np.asarray(ref_mat.mean(axis=0))
            bg_topic_dist = cosine_similarity(centroid_tfidf, ref_mat)[0] + 0.05
            d2v_vector = claim_topic_dist[:10].tolist()
            novelty_metrics = compute_novelty_metrics(claim_topic_dist, bg_topic_dist)
        else:
            novelty_metrics = compute_novelty_metrics(np.ones(100))

    # Node 9: Phân tích Cảm xúc (Surprise, Fear, Disgust, Trust), Phản bác (Dissent) & Heuristic
    emotion_heuristics = analyze_emotion_and_heuristics(effective_raw_text)

    # =================================================================
    # GIAI ĐOẠN 4 & 5 (Nodes 10, 11, 12): Đối chiếu Bằng chứng & Hợp nhất Phán quyết
    # =================================================================
    hybrid_verdict = synthesize_hybrid_verdict(
        claim_info=claim_analysis,
        retrieval_output=retrieval_output,
        ml_results=results,
        novelty_metrics=novelty_metrics,
        emotion_heuristics=emotion_heuristics,
    )
    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    hybrid_verdict["latency_ms"] = latency_ms

    return {
        "input_text": item.text,
        "cleaned_text": cleaned_text,
        "claim_analysis": claim_analysis,
        "hybrid_verdict": hybrid_verdict,
        "novelty_metrics": novelty_metrics,
        "emotion_heuristics": emotion_heuristics,
        "results": results,
        "latency_ms": latency_ms,
        "doc2vec_vector": d2v_vector[:10],
        "education_doc2vec": EDUCATIONAL_CONTENT["doc2vec"],
    }


@app.get("/benchmark-metrics")
def get_benchmark_metrics():
    bench_path = os.path.join(MODELS_DIR, "benchmark_metrics.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.post("/feedback")
def submit_user_feedback(fb: FeedbackItem):
    feedback_dir = os.path.join(BASE_DIR, "data", "feedback")
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, "user_feedback.json")

    existing = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    record = {
        "id": f"FB-{len(existing)+1:04d}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim": fb.claim[:500],
        "predicted_verdict": fb.predicted_verdict,
        "predicted_nli_label": fb.predicted_nli_label,
        "user_agreement": fb.user_agreement,
        "user_suggested_label": fb.user_suggested_label or fb.predicted_nli_label,
        "user_comment": fb.user_comment[:500] if fb.user_comment else "",
        "evidence_source_url": fb.evidence_source_url or "",
    }
    existing.insert(0, record)
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(existing[:200], f, ensure_ascii=False, indent=2)

    return {
        "status": "saved",
        "message": "Đã ghi nhận phản hồi của người dùng vào Kho kiểm duyệt Human-in-the-loop.",
        "feedback_id": record["id"],
        "total_feedback_count": len(existing),
    }


@app.get("/feedback")
def list_user_feedback():
    feedback_file = os.path.join(BASE_DIR, "data", "feedback", "user_feedback.json")
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            return {"count": len(items), "items": items[:20]}
        except Exception:
            pass
    return {"count": 0, "items": []}


if os.path.exists(os.path.join(BASE_DIR, "frontend", "dist")):
    app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "frontend", "dist"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
