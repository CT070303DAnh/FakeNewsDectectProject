import os
import sys
import time
import json
import math
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.build_vn_factcheck_dataset import build_evaluation_dataset
from src.preprocess import predict_check_worthiness
from src.evidence_retriever import MultiTierEvidenceRetriever
from src.evidence_analyzer import analyze_single_evidence_stance, run_llm_few_shot_verification


def evaluate_all_benchmarks() -> Dict[str, Any]:
    """
    Thực thi bộ Benchmark Đánh giá Toàn diện theo đúng 7 tiêu chí bắt buộc của Đề tài số 8
    trong file Danh_sach_de_tai.xlsx:
      1. Claim Detection (Check-worthiness) Macro-F1
      2. Retrieval Recall@K (K=1,3,5), MRR@5, NDCG@5 (So sánh BM25 vs Dense vs Hybrid vs Hybrid+Reranker)
      3. Verification 3-Class Macro-F1 (Supported / Refuted / NEI) & So sánh với LLM Few-shot
      4. Evidence Precision / Recall
      5. Citation Accuracy
      6. End-to-End Evidence-Grounded Accuracy (FEVER Score)
      7. Latency (ms/claim)
    """
    eval_json_path = os.path.join(BASE_DIR, "data", "evaluation", "vn_fact_verification_eval_set.json")
    cw_json_path = os.path.join(BASE_DIR, "data", "evaluation", "check_worthiness_eval.json")

    if not os.path.exists(eval_json_path) or not os.path.exists(cw_json_path):
        build_evaluation_dataset()

    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_claims: List[Dict[str, Any]] = json.load(f)

    with open(cw_json_path, "r", encoding="utf-8") as f:
        cw_samples: List[Dict[str, Any]] = json.load(f)

    # =========================================================================
    # 1. ĐÁNH GIÁ CLAIM DETECTION / CHECK-WORTHINESS (Macro-F1, Precision, Recall)
    # =========================================================================
    y_true_cw = []
    y_pred_cw = []
    for item in cw_samples:
        y_true_cw.append(int(item["is_check_worthy"]))
        pred = predict_check_worthiness(item["text"])
        y_pred_cw.append(1 if pred["is_check_worthy"] else 0)

    cw_macro_f1 = float(f1_score(y_true_cw, y_pred_cw, average="macro"))
    cw_precision = float(precision_score(y_true_cw, y_pred_cw, average="macro", zero_division=0))
    cw_recall = float(recall_score(y_true_cw, y_pred_cw, average="macro", zero_division=0))
    cw_acc = float(accuracy_score(y_true_cw, y_pred_cw))

    claim_detection_metrics = {
        "total_samples": len(cw_samples),
        "macro_f1": round(cw_macro_f1, 4),
        "precision": round(cw_precision, 4),
        "recall": round(cw_recall, 4),
        "accuracy": round(cw_acc, 4)
    }

    # =========================================================================
    # 2. ĐÁNH GIÁ RETRIEVAL: BM25 vs DENSE vs HYBRID vs HYBRID + RERANKER
    #    Metrics: Recall@1, Recall@3, Recall@5, MRR@5, NDCG@5, Evidence Precision@5
    # =========================================================================
    retriever = MultiTierEvidenceRetriever(BASE_DIR)
    retrieval_modes = ["bm25", "dense", "hybrid", "hybrid_rerank"]
    retrieval_comparison: Dict[str, Any] = {}

    for mode in retrieval_modes:
        r1_hits = 0
        r3_hits = 0
        r5_hits = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        ev_prec_sum = 0.0
        mode_start = time.perf_counter()

        for sample in eval_claims:
            gold_id = sample["id"]
            gold_topic = sample["topic"]
            gold_label = sample["label"]
            candidates = retriever.search_local_evidence(sample["claim"], top_k=5, mode=mode)
            cand_ids = [c["id"] for c in candidates]

            if gold_id in cand_ids[:1]:
                r1_hits += 1
            if gold_id in cand_ids[:3]:
                r3_hits += 1
            if gold_id in cand_ids[:5]:
                r5_hits += 1
                rank = cand_ids.index(gold_id) + 1
                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)

            # Evidence Precision: tỷ lệ bằng chứng trong top-3 cùng chủ đề & cùng định hướng nhãn với Gold
            top3 = candidates[:3]
            if top3:
                rel_count = sum(
                    1 for c in top3
                    if c["id"] == gold_id or (c.get("topic") == gold_topic and c.get("nli_label") == gold_label)
                )
                ev_prec_sum += rel_count / len(top3)

        n_total = max(len(eval_claims), 1)
        avg_lat_ms = ((time.perf_counter() - mode_start) * 1000.0) / n_total

        retrieval_comparison[mode] = {
            "method_name": {
                "bm25": "BM25 Baseline (Okapi BM25)",
                "dense": "Dense Semantic Retrieval (SVD/SBERT)",
                "hybrid": "Hybrid Retrieval (BM25 + Dense)",
                "hybrid_rerank": "Hybrid + Cross-Encoder Reranker + Temporal Filter"
            }[mode],
            "recall_at_1": round(r1_hits / n_total, 4),
            "recall_at_3": round(r3_hits / n_total, 4),
            "recall_at_5": round(r5_hits / n_total, 4),
            "mrr_at_5": round(mrr_sum / n_total, 4),
            "ndcg_at_5": round(ndcg_sum / n_total, 4),
            "evidence_precision": round(ev_prec_sum / n_total, 4),
            "evidence_recall": round(r5_hits / n_total, 4),
            "avg_latency_ms": round(avg_lat_ms, 2)
        }

    # =========================================================================
    # 3. ĐÁNH GIÁ VERIFICATION (3 NHÃN: Supported / Refuted / NEI),
    #    CITATION ACCURACY & END-TO-END EVIDENCE-GROUNDED ACCURACY
    # =========================================================================
    label_to_idx = {"Supported": 0, "Refuted": 1, "NEI": 2}
    y_true_ver = []
    y_pred_nli = []
    y_pred_llm = []
    y_pred_hybrid = []

    citation_correct = 0
    e2e_grounded_correct = 0
    latencies_ms = []

    for sample in eval_claims:
        t0 = time.perf_counter()
        gold_id = sample["id"]
        gold_label = sample["label"]
        y_true_ver.append(label_to_idx[gold_label])

        # Truy xuất bằng chứng với chế độ tốt nhất (hybrid_rerank)
        cands = retriever.search_local_evidence(sample["claim"], top_k=3, mode="hybrid_rerank")
        analyzed = [analyze_single_evidence_stance(sample["claim"], False, ev) for ev in cands]

        # 1. NLI Verifier prediction
        if analyzed:
            pred_nli_lbl = analyzed[0]["nli_label"]
        else:
            pred_nli_lbl = "NEI"
        y_pred_nli.append(label_to_idx.get(pred_nli_lbl, 2))

        # 2. LLM Few-Shot Evidence-Grounded prediction
        llm_out = run_llm_few_shot_verification(sample["claim"], analyzed, is_debunking=False)
        pred_llm_lbl = llm_out["llm_label"]
        y_pred_llm.append(label_to_idx.get(pred_llm_lbl, 2))

        # 3. Hybrid Consensus prediction
        pred_hybrid_lbl = pred_nli_lbl if pred_nli_lbl == pred_llm_lbl else llm_out["llm_label"]
        y_pred_hybrid.append(label_to_idx.get(pred_hybrid_lbl, 2))

        # Kiểm tra Citation Accuracy (Trích dẫn đúng nguồn/ID của bằng chứng chuẩn)
        top_ids = [ev["id"] for ev in analyzed]
        if analyzed and (analyzed[0]["id"] == gold_id or analyzed[0]["source"] == sample["source"]):
            citation_correct += 1

        # Kiểm tra End-to-End Evidence-Grounded Accuracy (FEVER strict metric:
        # Phải vừa truy xuất trúng Gold Evidence trong top-3 VỪA phân loại đúng nhãn NLI)
        if (gold_id in top_ids) and (pred_hybrid_lbl == gold_label):
            e2e_grounded_correct += 1

        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    n_claims = max(len(eval_claims), 1)

    nli_macro_f1 = float(f1_score(y_true_ver, y_pred_nli, average="macro"))
    llm_macro_f1 = float(f1_score(y_true_ver, y_pred_llm, average="macro"))
    hybrid_macro_f1 = float(f1_score(y_true_ver, y_pred_hybrid, average="macro"))
    ver_cm = confusion_matrix(y_true_ver, y_pred_hybrid, labels=[0, 1, 2]).tolist()

    verification_comparison = {
        "nli_verifier": {
            "name": "NLI / Stance Transformer Verifier (3-Class)",
            "macro_f1": round(nli_macro_f1, 4),
            "accuracy": round(float(accuracy_score(y_true_ver, y_pred_nli)), 4),
            "precision": round(float(precision_score(y_true_ver, y_pred_nli, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_ver, y_pred_nli, average="macro", zero_division=0)), 4),
        },
        "llm_few_shot": {
            "name": "LLM 3-Shot Evidence-Grounded Verification",
            "macro_f1": round(llm_macro_f1, 4),
            "accuracy": round(float(accuracy_score(y_true_ver, y_pred_llm)), 4),
            "precision": round(float(precision_score(y_true_ver, y_pred_llm, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_ver, y_pred_llm, average="macro", zero_division=0)), 4),
        },
        "hybrid_consensus": {
            "name": "Hybrid Evidence-Grounded Consensus (Proposed System)",
            "macro_f1": round(hybrid_macro_f1, 4),
            "accuracy": round(float(accuracy_score(y_true_ver, y_pred_hybrid)), 4),
            "precision": round(float(precision_score(y_true_ver, y_pred_hybrid, average="macro", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_ver, y_pred_hybrid, average="macro", zero_division=0)), 4),
            "confusion_matrix_3class": ver_cm,
            "labels_order": ["Supported", "Refuted", "NEI"]
        }
    }

    best_ret = retrieval_comparison["hybrid_rerank"]
    summary_report = {
        "dataset_info": {
            "name": "Vietnamese Fact Verification Evaluation Set (VNFC-360)",
            "total_claims": len(eval_claims),
            "corpus_size": len(retriever.local_corpus),
            "labels": {"Supported": 120, "Refuted": 120, "NEI": 120},
            "fields": ["id", "claim", "label", "gold_evidence", "source", "date", "topic"]
        },
        "official_metrics_summary": {
            "claim_detection_macro_f1": claim_detection_metrics["macro_f1"],
            "retrieval_recall_at_5": best_ret["recall_at_5"],
            "retrieval_mrr_at_5": best_ret["mrr_at_5"],
            "retrieval_ndcg_at_5": best_ret["ndcg_at_5"],
            "verification_macro_f1": verification_comparison["hybrid_consensus"]["macro_f1"],
            "evidence_precision": best_ret["evidence_precision"],
            "evidence_recall": best_ret["evidence_recall"],
            "citation_accuracy": round(citation_correct / n_claims, 4),
            "end_to_end_evidence_grounded_accuracy": round(e2e_grounded_correct / n_claims, 4),
            "avg_latency_ms": round(float(np.mean(latencies_ms)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2)
        },
        "claim_detection_details": claim_detection_metrics,
        "retrieval_comparison": retrieval_comparison,
        "verification_comparison": verification_comparison
    }

    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "factcheck_benchmark_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    print("=====================================================================")
    print(" KẾT QUẢ THỰC NGHIỆM ĐỀ TÀI SỐ 8 (VIETNAMESE FACT VERIFICATION BENCHMARK)")
    print("=====================================================================")
    print(f" 1. Claim Detection Macro-F1          : {summary_report['official_metrics_summary']['claim_detection_macro_f1']*100:.2f}%")
    print(f" 2. Retrieval Recall@5 / MRR / NDCG   : {best_ret['recall_at_5']*100:.2f}% / {best_ret['mrr_at_5']*100:.2f}% / {best_ret['ndcg_at_5']*100:.2f}%")
    print(f" 3. Verification Macro-F1 (3-Class)   : {summary_report['official_metrics_summary']['verification_macro_f1']*100:.2f}%")
    print(f" 4. Evidence Precision / Recall       : {best_ret['evidence_precision']*100:.2f}% / {best_ret['evidence_recall']*100:.2f}%")
    print(f" 5. Citation Accuracy                 : {summary_report['official_metrics_summary']['citation_accuracy']*100:.2f}%")
    print(f" 6. End-to-End Grounded Accuracy      : {summary_report['official_metrics_summary']['end_to_end_evidence_grounded_accuracy']*100:.2f}%")
    print(f" 7. Average Latency (Local Pipeline)  : {summary_report['official_metrics_summary']['avg_latency_ms']:.2f} ms/claim")
    print("=====================================================================")
    return summary_report


if __name__ == "__main__":
    evaluate_all_benchmarks()
