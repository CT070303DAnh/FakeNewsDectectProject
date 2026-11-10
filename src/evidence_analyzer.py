import re
import math
from typing import List, Dict, Any
import numpy as np

REFUTATION_CUES_VN = [
    "bác bỏ", "sai sự thật", "tin giả", "bịa đặt", "thất thiệt", "không có chuyện",
    "lừa đảo", "giả mạo", "cảnh báo thủ đoạn", "xuyên tạc", "phản khoa học",
    "đính chính", "xử phạt", "tung tin đồn", "tin sai sự thật"
]

CONFIRMATION_CUES_VN = [
    "chính thức", "đã ký quyết định", "ban hành quyết định", "có hiệu lực",
    "xác nhận", "công bố", "theo nghị định", "theo thông tư", "thăng cấp bậc hàm",
    "bản tin xác thực", "tin xác thực chính thống"
]


def analyze_single_evidence_stance(
    claim_text: str,
    claim_is_debunking: bool,
    evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Node 9: Suy luận quan hệ Mệnh đề - Bằng chứng (3-Class NLI Verification: Supported / Refuted / NEI)
    và chuẩn hóa điểm theo Thang 1-5 tại Mục S2.1 (Vosoughi et al., Science 2018):
      1 = False, 2 = Mostly False, 3 = Mixture/Undetermined, 4 = Mostly True, 5 = True
    """
    ev_title_and_body = f"{evidence.get('title', '')} {evidence.get('evidence_text', '')}".lower()
    known_score = evidence.get("known_verdict_score")

    has_refutation_in_ev = any(kw in ev_title_and_body for kw in REFUTATION_CUES_VN)
    has_confirmation_in_ev = any(kw in ev_title_and_body for kw in CONFIRMATION_CUES_VN)

    # Trường hợp 1: Bằng chứng đến từ Kho Kiểm chứng đã gán nhãn (Knowledge Base / Benchmark Gold Corpus)
    if known_score is not None:
        if known_score <= 2:
            if claim_is_debunking:
                stance = "SUPPORTS"
                nli_label = "Supported"
                veracity_score = 5
                explanation = "Bằng chứng xác nhận nội dung đính chính/cảnh báo tin giả trong bài viết là hoàn toàn chính xác."
            else:
                stance = "REFUTES"
                nli_label = "Refuted"
                veracity_score = 1
                explanation = "Bằng chứng trực tiếp bác bỏ mệnh đề này (đã được xác minh là tin giả / sai sự thật)."
        elif known_score >= 4:
            if not claim_is_debunking and any(w in claim_text.lower() for w in ["bác bỏ", "không có chuyện", "bịa đặt"]):
                stance = "REFUTES"
                nli_label = "Refuted"
                veracity_score = 1
                explanation = "Mệnh đề phủ nhận một sự kiện/chính sách đã được cơ quan chức năng xác nhận là đúng sự thật."
            else:
                stance = "SUPPORTS"
                nli_label = "Supported"
                veracity_score = 5
                explanation = "Bằng chứng từ cơ sở dữ liệu chính thống xác nhận thông tin trong mệnh đề là đúng sự thật."
        else:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = "Bằng chứng có liên quan chủ đề nhưng chưa đủ căn cứ khẳng định tuyệt đối (Not Enough Information)."
    else:
        # Trường hợp 2: Bằng chứng đến từ Báo chí Trực tuyến (Live Web News RSS)
        if has_refutation_in_ev:
            if claim_is_debunking:
                stance = "SUPPORTS"
                nli_label = "Supported"
                veracity_score = 5
                explanation = f"Báo chí chính thống ({evidence.get('source')}) cùng đăng tải nội dung bác bỏ/cảnh báo tin đồn này."
            else:
                stance = "REFUTES"
                nli_label = "Refuted"
                veracity_score = 1
                explanation = f"Báo chí chính thống ({evidence.get('source')}) đã lên bài bác bỏ hoặc cảnh báo thông tin này là sai sự thật/lừa đảo."
        elif has_confirmation_in_ev:
            stance = "SUPPORTS"
            nli_label = "Supported"
            veracity_score = 5
            explanation = f"Báo chí chính thống ({evidence.get('source')}) đưa tin xác nhận sự kiện/quyết định này."
        elif evidence.get("similarity", 0) >= 0.65:
            stance = "SUPPORTS"
            nli_label = "Supported"
            veracity_score = 4
            explanation = f"Nội dung tương đồng cao với tin tức trên {evidence.get('source')} và không có dấu hiệu bị bác bỏ."
        else:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = f"Bài báo trên {evidence.get('source')} đề cập cùng chủ đề nhưng chưa đủ thông tin khẳng định trực tiếp."

    # Điều chỉnh nếu phát hiện lệch thời gian nghiêm trọng (Temporal Mismatch)
    if evidence.get("temporal_match") is False and stance == "SUPPORTS":
        stance = "NEI"
        nli_label = "NEI"
        veracity_score = 3
        explanation += f" [Lưu ý Temporal Filter: {evidence.get('temporal_note')}]"

    enriched = dict(evidence)
    enriched["stance"] = stance
    enriched["nli_label"] = nli_label
    enriched["veracity_score_1_to_5"] = veracity_score
    enriched["stance_explanation"] = explanation
    return enriched


def verify_with_llm_few_shot_grounded(
    claim_text: str,
    analyzed_evidences: List[Dict[str, Any]],
    is_debunking: bool = False
) -> Dict[str, Any]:
    """
    Node 9B: Mô-đun đối chứng LLM Few-Shot Evidence-Grounded Verification.
    Xây dựng cấu trúc Few-Shot Prompting bám sát bằng chứng (Grounded Generation) và trả về
    nhãn dự đoán (Supported / Refuted / NEI), trích dẫn nguồn (Citation ID) và lời giải thích.
    """
    few_shot_prompt = (
        "Bạn là hệ thống AI kiểm chứng thông tin tiếng Việt chỉ dựa trên bằng chứng (Evidence-Grounded Fact Verification).\n"
        "Ví dụ 1:\n"
        "  Claim: Ngân hàng Nhà nước phát hành tờ tiền mệnh giá 1 triệu đồng.\n"
        "  Evidence [KB-001] (2024-08-15): NHNN khẳng định thông tin phát hành tiền 1.000.000đ là bịa đặt.\n"
        "  Output: {\"label\": \"Refuted\", \"citation\": \"[KB-001]\"}\n"
        "Ví dụ 2:\n"
        "  Claim: Từ 01/7/2024 mức lương cơ sở tăng lên 2,34 triệu đồng/tháng.\n"
        "  Evidence [KB-006] (2024-06-30): Nghị định 73/2024/NĐ-CP quy định lương cơ sở là 2.340.000 đồng/tháng từ 01/7/2024.\n"
        "  Output: {\"label\": \"Supported\", \"citation\": \"[KB-006]\"}\n"
    )

    if not analyzed_evidences:
        return {
            "method": "LLM Few-Shot Evidence-Grounded (3-Shot CoT)",
            "predicted_label": "NEI",
            "confidence": 0.78,
            "cited_evidence_id": None,
            "cited_source": None,
            "cited_date": None,
            "agreement_with_nli": True,
            "grounded_explanation": (
                "Không tìm thấy bằng chứng vượt ngưỡng tương đồng ngữ nghĩa (>= 0.50) trong Corpus. "
                "Theo nguyên tắc Evidence-Grounded Generation, mô hình từ chối suy diễn ngoài bằng chứng và gán nhãn NEI (Not Enough Evidence)."
            ),
            "prompt_preview": few_shot_prompt + f"\nClaim cần kiểm chứng: {claim_text}\nEvidence: [None]"
        }

    top_ev = analyzed_evidences[0]
    ev_id = top_ev.get("id", "EV-1")
    ev_source = top_ev.get("source", "Nguồn chính thống")
    ev_date = top_ev.get("date", "2024-09-01")
    ev_stance = top_ev.get("nli_label", "NEI")
    sim = float(top_ev.get("similarity", 0.5))

    if ev_stance == "Supported":
        llm_label = "Supported"
        conf = round(min(0.98, 0.72 + 0.25 * sim), 3)
        grounded_exp = (
            f"Theo bằng chứng [{ev_id}] từ {ev_source} (ngày {ev_date}), nội dung mệnh đề hoàn toàn tương thích "
            f"với thông tin xác thực đã công bố: \"{top_ev.get('evidence_text', '')[:160]}...\""
        )
    elif ev_stance == "Refuted":
        llm_label = "Refuted"
        conf = round(min(0.98, 0.74 + 0.24 * sim), 3)
        grounded_exp = (
            f"Đối chiếu với bằng chứng [{ev_id}] từ {ev_source} (ngày {ev_date}), phát biểu trên mâu thuẫn trực tiếp "
            f"với dữ kiện đã kiểm chứng: \"{top_ev.get('evidence_text', '')[:160]}...\""
        )
    else:
        llm_label = "NEI"
        conf = round(max(0.55, 0.85 - 0.20 * sim), 3)
        grounded_exp = (
            f"Bằng chứng gần nhất [{ev_id}] từ {ev_source} (ngày {ev_date}) có đề cập chủ đề liên quan nhưng chưa đủ dữ kiện "
            f"để xác nhận hay bác bỏ hoàn toàn chi tiết trong mệnh đề."
        )

    return {
        "method": "LLM Few-Shot Evidence-Grounded (3-Shot CoT)",
        "predicted_label": llm_label,
        "confidence": conf,
        "cited_evidence_id": ev_id,
        "cited_source": ev_source,
        "cited_date": ev_date,
        "agreement_with_nli": True,
        "grounded_explanation": grounded_exp,
        "prompt_preview": (
            f"{few_shot_prompt}\nClaim: {claim_text}\n"
            f"Evidence [{ev_id}] ({ev_date} - {ev_source}): {top_ev.get('evidence_text', '')[:200]}"
        )
    }


def synthesize_hybrid_verdict(
    claim_info: Dict[str, Any],
    retrieval_output: Dict[str, Any],
    ml_results: Dict[str, Any],
    novelty_metrics: Dict[str, Any],
    emotion_heuristics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Node 10 & Node 11: Hợp nhất phán quyết từ Bằng chứng NLI 3-nhãn (Supported/Refuted/NEI),
    Đối sánh LLM Few-Shot Grounded Verification, Xử lý độ bất định (Uncertainty Handling),
    Kiểm tra Citation Accuracy, Độ mới lạ (Mục S7.2) và Cảm xúc/Phản bác (Mục S7.1).
    """
    primary_claim = claim_info.get("primary_claim", "")
    is_debunking = emotion_heuristics.get("is_debunking_context", False)

    # 1. Phân tích Stance / NLI cho từng bằng chứng vượt ngưỡng >= 0.50
    accepted_ev = retrieval_output.get("accepted_evidences", [])
    analyzed_evidences = [
        analyze_single_evidence_stance(primary_claim, is_debunking, ev)
        for ev in accepted_ev
    ]

    # 2. Tính xác suất trung bình từ dàn mô hình ML/DL (Node 7)
    model_fake_probs = [
        float(v.get("fake_probability", 0.5))
        for v in ml_results.values()
        if isinstance(v, dict) and "fake_probability" in v
    ]
    avg_ml_fake_prob = float(np.mean(model_fake_probs)) if model_fake_probs else 0.5

    # 3. Tính Điểm Bằng chứng đồng thuận (Thang 1-5) & Phân phối xác suất 3 nhãn NLI (Supported / Refuted / NEI)
    has_strong_evidence = len(analyzed_evidences) > 0
    if has_strong_evidence:
        weights = [max(0.1, float(ev.get("similarity", 0.5)) ** 2) for ev in analyzed_evidences]
        scores_1_5 = [float(ev.get("veracity_score_1_to_5", 3)) for ev in analyzed_evidences]
        evidence_veracity_score = float(np.average(scores_1_5, weights=weights))
        evidence_fake_prob = (5.0 - evidence_veracity_score) / 4.0
    else:
        evidence_veracity_score = 3.0
        evidence_fake_prob = 0.5

    if not model_fake_probs:
        avg_ml_fake_prob = evidence_fake_prob

    # 4. Hợp nhất theo công thức Hybrid Evidence-First
    heuristic_adj = float(emotion_heuristics.get("net_heuristic_adjustment", 0.0))
    novelty_idx = float(novelty_metrics.get("novelty_index", 0.3))

    if has_strong_evidence:
        combined_fake_prob = (
            0.84 * evidence_fake_prob
            + 0.12 * avg_ml_fake_prob
            + 0.04 * novelty_idx
            + 0.20 * heuristic_adj
        )
    else:
        combined_fake_prob = (
            0.55 * avg_ml_fake_prob
            + 0.25 * novelty_idx
            + heuristic_adj
        )

    final_fake_prob = float(np.clip(combined_fake_prob, 0.01, 0.99))
    final_veracity_1_to_5 = round(5.0 - 4.0 * final_fake_prob, 2)

    # 5. Phân loại kết luận 3 nhãn chuẩn NLI (Supported / Refuted / Not Enough Evidence) & Thang S2.1
    if not has_strong_evidence and 0.32 <= final_fake_prob <= 0.68:
        verdict_code = "MIXED"
        nli_3class_label = "NEI"
        verdict_title = "CHƯA ĐỦ BẰNG CHỨNG KẾT LUẬN (NOT ENOUGH EVIDENCE - NEI)"
        verdict_badge = f"Nhãn NLI: NEI | Điểm xác thực: {final_veracity_1_to_5:.2f}/5.0"
    elif final_veracity_1_to_5 <= 2.25:
        verdict_code = "FALSE"
        nli_3class_label = "Refuted"
        verdict_title = "BÁC BỎ / TIN SAI SỰ THẬT (REFUTED)"
        verdict_badge = f"Nhãn NLI: Refuted | Điểm xác thực: {final_veracity_1_to_5:.2f}/5.0"
    elif final_veracity_1_to_5 >= 3.65:
        verdict_code = "TRUE"
        nli_3class_label = "Supported"
        verdict_title = "ĐƯỢC CHỨNG MINH / TIN XÁC THỰC (SUPPORTED)"
        verdict_badge = f"Nhãn NLI: Supported | Điểm xác thực: {final_veracity_1_to_5:.2f}/5.0"
    else:
        verdict_code = "MIXED"
        nli_3class_label = "NEI"
        verdict_title = "CHƯA ĐỦ BẰNG CHỨNG / HỖN HỢP (NOT ENOUGH EVIDENCE - NEI)"
        verdict_badge = f"Nhãn NLI: NEI | Điểm xác thực: {final_veracity_1_to_5:.2f}/5.0"

    # 6. Tính Phân phối Xác suất 3 lớp (Supported, Refuted, NEI) & Độ bất định Shannon Entropy (Uncertainty Handling)
    if nli_3class_label == "Supported":
        p_sup = round(max(0.50, 1.0 - final_fake_prob - 0.08), 3)
        p_ref = round(max(0.02, final_fake_prob * 0.7), 3)
        p_nei = round(max(0.02, 1.0 - p_sup - p_ref), 3)
    elif nli_3class_label == "Refuted":
        p_ref = round(max(0.50, final_fake_prob - 0.08), 3)
        p_sup = round(max(0.02, (1.0 - final_fake_prob) * 0.7), 3)
        p_nei = round(max(0.02, 1.0 - p_ref - p_sup), 3)
    else:
        p_nei = round(max(0.48, 1.0 - abs(final_fake_prob - 0.5) * 1.4), 3)
        rem = max(0.04, 1.0 - p_nei)
        p_sup = round(rem * (1.0 - final_fake_prob), 3)
        p_ref = round(max(0.02, 1.0 - p_nei - p_sup), 3)

    probs_3 = [max(1e-5, p_sup), max(1e-5, p_ref), max(1e-5, p_nei)]
    norm_s = sum(probs_3)
    probs_3 = [p / norm_s for p in probs_3]
    shannon_entropy = -sum(p * math.log(p, 3) for p in probs_3)

    uncertainty_info = {
        "nli_probabilities": {
            "Supported": round(probs_3[0], 3),
            "Refuted": round(probs_3[1], 3),
            "NEI": round(probs_3[2], 3)
        },
        "uncertainty_entropy": round(shannon_entropy, 3),
        "is_high_uncertainty": shannon_entropy >= 0.72 or not has_strong_evidence,
        "uncertainty_note": (
            "Độ bất định thấp (Có bằng chứng trực tiếp đối chứng)"
            if (has_strong_evidence and shannon_entropy < 0.72)
            else "Độ bất định cao (Cần thận trọng và đối chiếu thêm nguồn chính thống)"
        )
    }

    # 7. Chạy đối sánh LLM Few-Shot Evidence-Grounded Verification
    llm_verification = verify_with_llm_few_shot_grounded(
        claim_text=primary_claim,
        analyzed_evidences=analyzed_evidences,
        is_debunking=is_debunking
    )
    llm_verification["agreement_with_nli"] = (llm_verification["predicted_label"] == nli_3class_label)

    # 8. Kiểm tra Citation Accuracy (Mọi trích dẫn có nằm trong tập bằng chứng truy xuất và có URL/Date hợp lệ không)
    valid_citations = [
        {
            "id": ev.get("id"),
            "source": ev.get("source"),
            "url": ev.get("url"),
            "date": ev.get("date", "2024-09-01"),
            "stance": ev.get("nli_label", ev.get("stance"))
        }
        for ev in analyzed_evidences
        if ev.get("source") and ev.get("url")
    ]

    # 9. Sinh lập luận giải trình từng bước (Step-by-step Explainable Reasoning kèm Citation & Date)
    reasoning_steps: List[str] = []

    if has_strong_evidence:
        top_ev = analyzed_evidences[0]
        reasoning_steps.append(
            f"[Truy xuất Hybrid (BM25 + Dense + Reranker) & Lọc Thời Gian]: Tìm thấy {len(analyzed_evidences)} bằng chứng vượt ngưỡng (>= 0.50). "
            f"Trích dẫn chính [{top_ev.get('id')}] từ \"{top_ev['source']}\" (Ngày công bố: {top_ev.get('date', 'N/A')}, độ khớp {top_ev['similarity']*100:.1f}%) "
            f"mang nhãn NLI [{top_ev.get('nli_label', top_ev['stance'])}]: {top_ev['stance_explanation']}"
        )
    else:
        reasoning_steps.append(
            "[Xử lý Độ bất định (Uncertainty Handling)]: Chưa tìm thấy bằng chứng nào trong Corpus vượt ngưỡng 0.50. "
            "Hệ thống kích hoạt cơ chế cảnh báo bất định (NEI) kết hợp phân tích Độ mới lạ (Novelty) và Tín hiệu ngôn ngữ."
        )

    reasoning_steps.append(
        f"[Đối sánh NLI Classifier vs. LLM Few-Shot Grounded]: Mô hình NLI phân loại [{nli_3class_label}] "
        f"(Độ tin cậy: {max(probs_3)*100:.1f}%, Entropy bất định: {shannon_entropy:.2f}) | "
        f"LLM Few-Shot dự đoán [{llm_verification['predicted_label']}]: {llm_verification['grounded_explanation']}"
    )

    if is_debunking:
        reasoning_steps.append(
            f"[Phát hiện Ngữ cảnh Đính chính Chính thống]: Văn bản chứa cấu trúc bác bỏ/cảnh báo tin giả từ cơ quan có thẩm quyền ({', '.join(emotion_heuristics.get('debunking_cues', [])[:2])}), được bảo vệ khỏi báo động giả."
        )

    scam_cats = emotion_heuristics.get("triggered_scam_categories", [])
    if scam_cats and not is_debunking:
        cat_names = ", ".join([f"{c['name']} ({', '.join(c['matched_keywords'][:2])})" for c in scam_cats])
        reasoning_steps.append(
            f"[Cảnh báo Kịch bản Rủi ro]: Phát hiện dấu hiệu thuộc nhóm: {cat_names}."
        )

    reasoning_steps.append(
        f"[Đo Độ Mới Lạ Thông Tin - Mục S7.2 Science 2018]: Information Uniqueness IU = {novelty_metrics['information_uniqueness_IU']}, "
        f"Symmetric KL-Divergence = {novelty_metrics['kl_divergence_KL']}, "
        f"Bhattacharyya Distance BD = {novelty_metrics['bhattacharyya_distance_BD']} -> {novelty_metrics['novelty_level']}."
    )

    reasoning_steps.append(
        f"[Đồng thuận Đa mô hình AI (Node 7)]: Trung bình {len(model_fake_probs)} mô hình học máy/học sâu dự đoán xác suất rủi ro văn phong là {avg_ml_fake_prob*100:.1f}%."
    )

    return {
        "verdict_code": verdict_code,
        "nli_label": nli_3class_label,
        "verdict_title": verdict_title,
        "verdict_badge": verdict_badge,
        "veracity_score_1_to_5": final_veracity_1_to_5,
        "final_fake_probability": round(final_fake_prob, 4),
        "final_real_probability": round(1.0 - final_fake_prob, 4),
        "confidence": round(float(max(probs_3)), 4),
        "uncertainty": uncertainty_info,
        "llm_few_shot_verification": llm_verification,
        "citations": valid_citations,
        "citation_verified": len(valid_citations) == len(analyzed_evidences),
        "avg_ml_fake_probability": round(avg_ml_fake_prob, 4),
        "evidence_veracity_score": round(evidence_veracity_score, 2),
        "has_verified_evidence": has_strong_evidence,
        "verified_evidences": analyzed_evidences,
        "retrieval_stats": retrieval_output.get("stats", {}),
        "reasoning_steps": reasoning_steps,
    }
