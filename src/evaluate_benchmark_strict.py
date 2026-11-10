"""
ĐÁNH GIÁ CHẶT (STRICT / LEAKAGE-FREE BENCHMARK)
================================================
Khác biệt so với src/evaluate_benchmark.py:
  1. Kho bằng chứng được lập chỉ mục ở chế độ evidence_only:
       - Chỉ chứa title + gold_evidence (KHÔNG chứa claim / claim_pattern của bộ đánh giá).
       - KHÔNG mang nhãn có sẵn (label / verdict_score) -> Verifier phải suy luận từ nội dung bằng chứng.
       - KHÔNG nạp mẫu ReINTEL (các mẫu này ghi sẵn nhãn trong evidence_text).
  2. Citation Accuracy tính chặt: top-1 phải đúng ID bằng chứng chuẩn (không chấp nhận "cùng nguồn").
  3. Bổ sung thực nghiệm "Oracle Evidence": đưa thẳng gold_evidence cho Verifier để tách riêng
     chất lượng khâu Verification khỏi khâu Retrieval.

Kết quả ghi ra: models/factcheck_benchmark_metrics_strict.json
(không ghi đè models/factcheck_benchmark_metrics.json của bản đánh giá cũ).
"""
import os
import sys
import time
import json
import math
from collections import Counter
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.evidence_retriever import MultiTierEvidenceRetriever
from src.evidence_analyzer import analyze_single_evidence_stance, run_llm_few_shot_verification

LABELS = ["Supported", "Refuted", "NEI"]
L2I = {l: i for i, l in enumerate(LABELS)}


def _cls_report(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "f1_per_class": {LABELS[i]: round(float(per_class[i]), 4) for i in range(3)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "labels_order": LABELS,
    }


def _verify(claim: str, evidences: List[Dict[str, Any]]) -> Dict[str, str]:
    analyzed = [analyze_single_evidence_stance(claim, False, ev) for ev in evidences]
    nli = analyzed[0]["nli_label"] if analyzed else "NEI"
    llm = run_llm_few_shot_verification(claim, analyzed, is_debunking=False)["llm_label"]
    hybrid = nli if nli == llm else llm
    return {"nli": nli, "llm": llm, "hybrid": hybrid, "analyzed": analyzed}


def evaluate_strict() -> Dict[str, Any]:
    eval_path = os.path.join(BASE_DIR, "data", "evaluation", "vn_fact_verification_eval_set.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_claims: List[Dict[str, Any]] = json.load(f)

    retriever = MultiTierEvidenceRetriever(BASE_DIR, evidence_only=True, include_historical=False)
    id2doc = {d["id"]: d for d in retriever.local_corpus}

    # Kiểm tra rò rỉ: không claim nào của bộ đánh giá được xuất hiện nguyên văn trong kho
    leaked = sum(1 for s in eval_claims if s["claim"] and any(s["claim"] in d["search_blob"] for d in retriever.local_corpus))

    # Thống kê mức độ "sinh theo mẫu" của bộ dữ liệu (để ghi chú trong báo cáo)
    template_ids = Counter((s["topic"], s["label"]) for s in eval_claims if str(s["id"]).startswith("VNFC-"))

    # ------------------------------------------------------------------
    # 1. RETRIEVAL: BM25 vs DENSE vs HYBRID vs HYBRID + RERANK
    # ------------------------------------------------------------------
    retrieval: Dict[str, Any] = {}
    for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        r1 = r3 = r5 = 0
        mrr = ndcg = 0.0
        same_topic_wrong_doc = 0
        t0 = time.perf_counter()
        for s in eval_claims:
            cands = retriever.search_local_evidence(s["claim"], top_k=5, mode=mode)
            ids = [c["id"] for c in cands]
            if s["id"] in ids[:1]:
                r1 += 1
            elif cands and cands[0].get("topic") == s["topic"]:
                # Lấy nhầm bằng chứng "anh em" cùng chủ đề (thường chỉ khác con số / tháng / năm)
                same_topic_wrong_doc += 1
            if s["id"] in ids[:3]:
                r3 += 1
            if s["id"] in ids[:5]:
                r5 += 1
                rank = ids.index(s["id"]) + 1
                mrr += 1.0 / rank
                ndcg += 1.0 / math.log2(rank + 1)
        n = max(len(eval_claims), 1)
        retrieval[mode] = {
            "recall_at_1": round(r1 / n, 4),
            "recall_at_3": round(r3 / n, 4),
            "recall_at_5": round(r5 / n, 4),
            "mrr_at_5": round(mrr / n, 4),
            "ndcg_at_5": round(ndcg / n, 4),
            "top1_wrong_but_same_topic": round(same_topic_wrong_doc / n, 4),
            "avg_latency_ms": round((time.perf_counter() - t0) * 1000.0 / n, 2),
        }

    # ------------------------------------------------------------------
    # 2. VERIFICATION (pipeline thật: evidence truy xuất bởi hybrid_rerank)
    # 3. VERIFICATION (oracle: đưa thẳng gold evidence)
    # ------------------------------------------------------------------
    y_true: List[int] = []
    pred = {"nli": [], "llm": [], "hybrid": []}
    pred_oracle = {"nli": [], "llm": [], "hybrid": []}
    citation_ok = 0
    e2e_ok = 0
    latencies: List[float] = []

    for s in eval_claims:
        t0 = time.perf_counter()
        y_true.append(L2I[s["label"]])

        cands = retriever.search_local_evidence(s["claim"], top_k=3, mode="hybrid_rerank")
        out = _verify(s["claim"], cands)
        for k in pred:
            pred[k].append(L2I.get(out[k], 2))

        top_ids = [c["id"] for c in cands]
        if top_ids and top_ids[0] == s["id"]:
            citation_ok += 1
        if s["id"] in top_ids and out["hybrid"] == s["label"]:
            e2e_ok += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)

        gold_doc = dict(id2doc.get(s["id"], {}))
        if gold_doc:
            gold_doc["similarity"] = 0.95
            out_o = _verify(s["claim"], [gold_doc])
        else:
            out_o = {"nli": "NEI", "llm": "NEI", "hybrid": "NEI"}
        for k in pred_oracle:
            pred_oracle[k].append(L2I.get(out_o[k], 2))

    n = max(len(eval_claims), 1)
    report = {
        "protocol": {
            "name": "Strict Leakage-free Evaluation",
            "index_fields": ["title", "gold_evidence"],
            "labels_hidden_from_verifier": True,
            "historical_reintel_in_corpus": False,
            "eval_claims_found_verbatim_in_corpus": leaked,
            "corpus_size": len(retriever.local_corpus),
            "total_claims": len(eval_claims),
            "label_distribution": dict(Counter(s["label"] for s in eval_claims)),
            "template_generated_claims": int(sum(template_ids.values())),
            "distinct_templates": len(template_ids),
        },
        "retrieval_comparison": retrieval,
        "verification_pipeline": {k: _cls_report(y_true, v) for k, v in pred.items()},
        "verification_oracle_evidence": {k: _cls_report(y_true, v) for k, v in pred_oracle.items()},
        "summary": {
            "retrieval_recall_at_5": retrieval["hybrid_rerank"]["recall_at_5"],
            "retrieval_mrr_at_5": retrieval["hybrid_rerank"]["mrr_at_5"],
            "verification_macro_f1": _cls_report(y_true, pred["hybrid"])["macro_f1"],
            "citation_accuracy_strict": round(citation_ok / n, 4),
            "end_to_end_evidence_grounded_accuracy": round(e2e_ok / n, 4),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
        },
    }

    out_path = os.path.join(BASE_DIR, "models", "factcheck_benchmark_metrics_strict.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    p = report["protocol"]
    print("=" * 72)
    print(" STRICT LEAKAGE-FREE BENCHMARK")
    print("=" * 72)
    print(f" Corpus: {p['corpus_size']} docs | Claims: {p['total_claims']} | Leaked verbatim: {p['eval_claims_found_verbatim_in_corpus']}")
    print(f" Template-generated claims: {p['template_generated_claims']} from {p['distinct_templates']} templates")
    print("-" * 72)
    print(f" {'Mode':<15}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR@5':>8}{'NDCG@5':>8}{'WrongSib':>10}")
    for m, r in retrieval.items():
        print(f" {m:<15}{r['recall_at_1']:>8.3f}{r['recall_at_3']:>8.3f}{r['recall_at_5']:>8.3f}"
              f"{r['mrr_at_5']:>8.3f}{r['ndcg_at_5']:>8.3f}{r['top1_wrong_but_same_topic']:>10.3f}")
    print("-" * 72)
    for title, block in [("Pipeline (retrieved)", report["verification_pipeline"]),
                         ("Oracle (gold evidence)", report["verification_oracle_evidence"])]:
        print(f" Verification - {title}")
        for k, r in block.items():
            print(f"   {k:<7} Macro-F1={r['macro_f1']:.3f}  Acc={r['accuracy']:.3f}  per-class={r['f1_per_class']}")
    print("-" * 72)
    for k, v in report["summary"].items():
        print(f" {k:<40}: {v}")
    print("=" * 72)
    return report


if __name__ == "__main__":
    evaluate_strict()

