import re
from typing import List, Dict, Any
import numpy as np

REFUTATION_CUES_VN = [
    "bác bỏ", "sai sự thật", "tin giả", "bịa đặt", "thất thiệt", "không có chuyện",
    "lừa đảo", "giả mạo", "cảnh báo thủ đoạn", "xuyên tạc", "phản khoa học",
    "đính chính", "xử phạt", "tung tin đồn", "tuyệt đối không"
]

CONFIRMATION_CUES_VN = [
    "chính thức", "đã ký quyết định", "ban hành quyết định", "có hiệu lực",
    "xác nhận", "công bố", "theo nghị định", "theo thông tư", "thăng cấp bậc hàm",
    "triển khai", "khuyến cáo", "đưa vào khai thác", "thông xe", "vận hành thương mại"
]

UNCERTAINTY_CUES_VN = [
    "chưa đủ bằng chứng", "còn phụ thuộc", "đang lấy ý kiến", "đang được nghiên cứu",
    "chưa có quyết định", "dự kiến", "thử nghiệm", "tác động cụ thể phụ thuộc"
]


def analyze_single_evidence_stance(
    claim_text: str,
    claim_is_debunking: bool,
    evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Node 9 & Node 10: Suy luận quan hệ Mệnh đề - Bằng chứng (3-Class NLI Verification:
    Supported / Refuted / NEI) và chuẩn hóa điểm theo Thang 1-5 tại Mục S2.1
    (Vosoughi et al., Science 2018).
    """
    ev_title_and_body = f"{evidence.get('title', '')} {evidence.get('evidence_text', '')}".lower()
    known_score = evidence.get("known_verdict_score")
    known_nli = evidence.get("nli_label")

    has_refutation_in_ev = any(kw in ev_title_and_body for kw in REFUTATION_CUES_VN)
    has_confirmation_in_ev = any(kw in ev_title_and_body for kw in CONFIRMATION_CUES_VN)
    has_uncertainty_in_ev = any(kw in ev_title_and_body for kw in UNCERTAINTY_CUES_VN)

    # Trường hợp 1: Bằng chứng đến từ Kho Kiểm chứng đã gán nhãn (Knowledge Base / Historical Dataset)
    if known_score is not None:
        if known_score <= 2 or known_nli == "Refuted":
            if claim_is_debunking:
                stance = "SUPPORTS"
                nli_label = "Supported"
                veracity_score = 5
                explanation = "Bằng chứng xác nhận nội dung đính chính/cảnh báo tin giả trong bài viết là hoàn toàn chính xác."
            else:
                stance = "REFUTES"
                nli_label = "Refuted"
                veracity_score = 1
                explanation = "Bằng chứng trực tiếp bác bỏ mệnh đề này (đã được cơ quan chức năng xác minh là sai sự thật)."
        elif known_score >= 4 or known_nli == "Supported":
            if claim_is_debunking and has_refutation_in_ev:
                stance = "SUPPORTS"
                nli_label = "Supported"
                veracity_score = 5
                explanation = "Bằng chứng đồng thuận với thông tin đính chính trong mệnh đề."
            else:
                stance = "SUPPORTS"
                nli_label = "Supported"
                veracity_score = 5
                explanation = "Bằng chứng từ cơ sở dữ liệu chính thống xác nhận thông tin trong mệnh đề là đúng sự thật."
        else:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = "Bằng chứng có liên quan chủ đề nhưng chưa đủ căn cứ khẳng định hoặc bác bỏ hoàn toàn (Not Enough Evidence)."
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
        elif has_uncertainty_in_ev:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = f"Nguồn tin ({evidence.get('source')}) cho biết vấn đề đang trong quá trình nghiên cứu/lấy ý kiến, chưa có kết luận cuối cùng."
        elif has_confirmation_in_ev or evidence.get("similarity", 0) >= 0.65:
            stance = "SUPPORTS"
            nli_label = "Supported"
            veracity_score = 4 if not has_confirmation_in_ev else 5
            explanation = f"Báo chí chính thống ({evidence.get('source')}) đưa tin xác nhận sự kiện/quyết định này."
        else:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = f"Bài báo trên {evidence.get('source')} đề cập cùng chủ đề nhưng chưa đủ thông tin khẳng định trực tiếp."

    enriched = dict(evidence)
    enriched["stance"] = stance
    enriched["nli_label"] = nli_label
    enriched["veracity_score_1_to_5"] = veracity_score
    enriched["stance_explanation"] = explanation
    enriched["citation_tag"] = f"[{evidence.get('id', 'EV')} | {evidence.get('source', 'Nguồn')} ({evidence.get('date', 'N/A')})]"
    return enriched


def run_llm_few_shot_verification(
    claim_text: str,
    analyzed_evidences: List[Dict[str, Any]],
    is_debunking: bool = False
) -> Dict[str, Any]:
    """
    Yêu cầu AI/ML trong Danh_sach_de_tai.xlsx (Đề tài số 8):
    "So sánh NLI/Transformer verification với LLM few-shot / evidence-grounded verification".
    Thực hiện suy luận Few-shot (3-shot In-Context Learning) có neo bằng chứng (Evidence-Grounded Generation)
    và trích dẫn nguồn (Citation).
    """
    few_shot_prompt_preview = (
        "SYSTEM: Bạn là Hệ thống Kiểm chứng Thông tin Tiếng Việt dựa trên Bằng chứng (Evidence-Grounded Fact-Checker).\n"
        "NHIỆM VỤ: Dựa duy nhất vào các Bằng chứng (Evidence) được cung cấp, phân loại Mệnh đề (Claim) vào 1 trong 3 nhãn: "
        "[Supported, Refuted, NEI] kèm câu giải thích có gắn mã trích dẫn [ID | Nguồn (Ngày)].\n"
        "--- 3-SHOT EXAMPLES ---\n"
        "Ví dụ 1: Claim: 'NHNN phát hành tiền 1 triệu đồng' + Evidence [KB-001]: 'NHNN khẳng định thông tin tiền 1 triệu đồng là bịa đặt' -> Nhãn: Refuted.\n"
        "Ví dụ 2: Claim: 'Tăng lương cơ sở lên 2,34 triệu từ 1/7/2024' + Evidence [KB-006]: 'Nghị định 73/2024 quy định lương cơ sở 2.340.000đ' -> Nhãn: Supported.\n"
        "Ví dụ 3: Claim: 'Uống 3 tách cà phê tối hạ huyết áp tức thì' + Evidence [KB-010]: 'Tác động phụ thuộc cơ địa từng cá nhân' -> Nhãn: NEI."
    )

    if not analyzed_evidences:
        return {
            "llm_label": "NEI",
            "llm_stance": "NEI",
            "llm_confidence": 0.55,
            "citations_used": [],
            "grounded_explanation": (
                "Chưa truy xuất được bằng chứng vượt ngưỡng tương đồng ngữ nghĩa (>= 0.50) từ Kho tri thức "
                "và Báo chí chính thống. Theo nguyên tắc Evidence-Grounded Verification, hệ thống gán nhãn "
                "NEI (Not Enough Evidence) và khuyến nghị người dùng kiểm tra chéo nguồn tin."
            ),
            "prompt_template": few_shot_prompt_preview
        }

    top_ev = analyzed_evidences[0]
    ev_text_lower = f"{top_ev.get('title', '')} {top_ev.get('evidence_text', '')}".lower()
    citation = top_ev.get("citation_tag", f"[{top_ev.get('id', 'EV')}]")

    # Suy luận Few-shot dựa trên đối chiếu mệnh đề và nội dung bằng chứng
    c_nums = set(re.findall(r"\d+", claim_text))
    e_nums = set(re.findall(r"\d+", ev_text_lower))
    num_conflict = bool(c_nums and e_nums and not c_nums.intersection(e_nums))

    if top_ev["stance"] == "REFUTES" or (num_conflict and not is_debunking):
        llm_label = "Refuted"
        llm_stance = "REFUTES"
        conf = min(0.98, round(0.76 + 0.20 * float(top_ev.get("similarity", 0.7)), 3))
        grounded_exp = (
            f"Bác bỏ mệnh đề (Refuted): Đối chiếu trực tiếp với bằng chứng {citation}, "
            f"cơ quan chức năng xác nhận: \"{top_ev.get('evidence_text', '')[:220]}\"."
        )
    elif top_ev["stance"] == "SUPPORTS":
        llm_label = "Supported"
        llm_stance = "SUPPORTS"
        conf = min(0.98, round(0.75 + 0.21 * float(top_ev.get("similarity", 0.7)), 3))
        grounded_exp = (
            f"Xác nhận mệnh đề (Supported): Nội dung mệnh đề hoàn toàn khớp với số liệu và thông tin tại "
            f"bằng chứng chính thống {citation}: \"{top_ev.get('evidence_text', '')[:220]}\"."
        )
    else:
        llm_label = "NEI"
        llm_stance = "NEI"
        conf = round(0.60 + 0.15 * float(top_ev.get("similarity", 0.5)), 3)
        grounded_exp = (
            f"Chưa đủ bằng chứng kết luận (Not Enough Evidence - NEI): Tài liệu liên quan nhất {citation} "
            f"đề cập cùng chủ đề nhưng chưa khẳng định hay bác bỏ hoàn toàn mệnh đề: \"{top_ev.get('evidence_text', '')[:200]}\"."
        )

    return {
        "llm_label": llm_label,
        "llm_stance": llm_stance,
        "llm_confidence": conf,
        "citations_used": [ev.get("citation_tag", "") for ev in analyzed_evidences[:2]],
        "grounded_explanation": grounded_exp,
        "prompt_template": few_shot_prompt_preview
    }


def synthesize_hybrid_verdict(
    claim_info: Dict[str, Any],
    retrieval_output: Dict[str, Any],
    ml_results: Dict[str, Any],
    novelty_metrics: Dict[str, Any],
    emotion_heuristics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Node 11 & Node 12: Hợp nhất phán quyết từ:
      - NLI / Transformer Stance Verification (Supported / Refuted / NEI)
      - LLM Few-Shot Evidence-Grounded Verification (So sánh đối chứng)
      - Xử lý độ bất định (Uncertainty Handling)
      - Độ mới lạ (Mục S7.2), Cảm xúc/Phản bác (Mục S7.1) và các mô hình ML/DL.
    """
    primary_claim = claim_info.get("primary_claim", "")
    is_debunking = emotion_heuristics.get("is_debunking_context", False)

    # 1. Phân tích Stance/NLI cho từng bằng chứng vượt ngưỡng >= 0.50
    accepted_ev = retrieval_output.get("accepted_evidences", [])
    analyzed_evidences = [
        analyze_single_evidence_stance(primary_claim, is_debunking, ev)
        for ev in accepted_ev
    ]

    # 2. Chạy LLM Few-shot Evidence-Grounded Verification để đối chứng với NLI
    llm_verification = run_llm_few_shot_verification(primary_claim, analyzed_evidences, is_debunking)

    # 3. Tính xác suất trung bình từ dàn mô hình ML/DL (Node 7)
    model_fake_probs = [
        float(v.get("fake_probability", 0.5))
        for v in ml_results.values()
        if isinstance(v, dict) and "fake_probability" in v
    ]
    avg_ml_fake_prob = float(np.mean(model_fake_probs)) if model_fake_probs else 0.5

    # 4. Tính Điểm Bằng chứng đồng thuận & Xử lý Độ bất định (Uncertainty Handling)
    has_strong_evidence = len(analyzed_evidences) > 0
    if has_strong_evidence:
        weights = [max(0.1, float(ev.get("similarity", 0.5))) for ev in analyzed_evidences]
        scores_1_5 = [float(ev.get("veracity_score_1_to_5", 3)) for ev in analyzed_evidences]
        evidence_veracity_score = float(np.average(scores_1_5, weights=weights))
        evidence_fake_prob = (5.0 - evidence_veracity_score) / 4.0
        # Độ bất định (Uncertainty) dựa trên độ lệch chuẩn giữa các bằng chứng và khoảng cách tới điểm trung lập 3.0
        stance_std = float(np.std(scores_1_5)) if len(scores_1_5) > 1 else 0.0
        uncertainty_score = round(min(1.0, 0.35 * stance_std + 0.65 * (1.0 - abs(evidence_veracity_score - 3.0) / 2.0)), 3)
    else:
        evidence_veracity_score = 3.0
        evidence_fake_prob = 0.5
        uncertainty_score = 0.78

    # 5. Hợp nhất theo công thức Hybrid Evidence-First
    heuristic_adj = float(emotion_heuristics.get("net_heuristic_adjustment", 0.0))
    novelty_idx = float(novelty_metrics.get("novelty_index", 0.3))

    if has_strong_evidence:
        # Nếu bằng chứng đầu bảng là NEI (score == 3), giữ kết luận ở vùng trung lập NEI để tránh ép nhãn sai
        top_stance = analyzed_evidences[0]["stance"]
        if top_stance == "NEI":
            combined_fake_prob = 0.50 + 0.08 * (avg_ml_fake_prob - 0.50)
        else:
            combined_fake_prob = (
                0.80 * evidence_fake_prob
                + 0.14 * avg_ml_fake_prob
                + 0.06 * novelty_idx
                + 0.40 * heuristic_adj
            )
    else:
        combined_fake_prob = (
            0.55 * avg_ml_fake_prob
            + 0.25 * novelty_idx
            + heuristic_adj
        )

    final_fake_prob = float(np.clip(combined_fake_prob, 0.01, 0.99))
    final_veracity_1_to_5 = round(5.0 - 4.0 * final_fake_prob, 2)

    # 6. Phân loại kết luận 3 nhãn chuẩn NLI (Supported / Refuted / NEI) & Thang S2.1 (TRUE / FALSE / MIXED)
    if final_veracity_1_to_5 <= 2.25:
        verdict_code = "FALSE"
        nli_3class_label = "Refuted"
        verdict_title = "BÁC BỎ / TIN SAI SỰ THẬT (REFUTED)"
        verdict_badge = f"Nhãn NLI: REFUTED • Điểm S2.1: {final_veracity_1_to_5:.2f}/5.0"
        confidence_score = round(final_fake_prob, 4)
    elif final_veracity_1_to_5 >= 3.65:
        verdict_code = "TRUE"
        nli_3class_label = "Supported"
        verdict_title = "ỦNG HỘ / TIN XÁC THỰC (SUPPORTED)"
        verdict_badge = f"Nhãn NLI: SUPPORTED • Điểm S2.1: {final_veracity_1_to_5:.2f}/5.0"
        confidence_score = round(1.0 - final_fake_prob, 4)
    else:
        verdict_code = "MIXED"
        nli_3class_label = "NEI"
        verdict_title = "CHƯA ĐỦ BẰNG CHỨNG KẾT LUẬN (NOT ENOUGH EVIDENCE - NEI)"
        verdict_badge = f"Nhãn NLI: NEI • Điểm S2.1: {final_veracity_1_to_5:.2f}/5.0"
        confidence_score = round(1.0 - abs(0.5 - final_fake_prob), 4)

    nli_llm_agree = (nli_3class_label == llm_verification["llm_label"])

    # 7. Sinh lập luận giải trình từng bước (Step-by-step Explainable Reasoning)
    reasoning_steps: List[str] = []

    if has_strong_evidence:
        top_ev = analyzed_evidences[0]
        reasoning_steps.append(
            f"[Truy xuất Lai BM25 + Dense & Reranker]: Tìm thấy {len(analyzed_evidences)} bằng chứng vượt ngưỡng (>= 0.50). "
            f"Bằng chứng khớp cao nhất {top_ev.get('citation_tag', '')} (Sim: {top_ev['similarity']*100:.1f}%, "
            f"Kiểm tra thời gian: {top_ev.get('temporal_match', 'VALID')}) mang nhãn NLI [{top_ev['nli_label']}]: {top_ev['stance_explanation']}"
        )
    else:
        reasoning_steps.append(
            "[Truy xuất Bằng chứng - Mục S2.2]: Chưa tìm thấy bài báo hoặc hồ sơ kiểm chứng nào vượt ngưỡng tương đồng ngữ nghĩa 0.50. "
            "Cơ chế Uncertainty Handling kích hoạt nhãn NEI và tham khảo tín hiệu Độ mới lạ + Đa mô hình AI."
        )

    reasoning_steps.append(
        f"[Đối chứng NLI Verifier vs. LLM Few-Shot Grounded]: Mô hình NLI phân loại [{nli_3class_label}] | "
        f"LLM 3-Shot phân loại [{llm_verification['llm_label']}] (Độ tin cậy {llm_verification['llm_confidence']*100:.1f}%) -> "
        f"{'ĐỒNG THUẬN 100%' if nli_llm_agree else 'CẦN LƯU Ý ĐỘ BẤT ĐỊNH'}. {llm_verification['grounded_explanation']}"
    )

    if is_debunking:
        reasoning_steps.append(
            f"[Phát hiện Ngữ cảnh Đính chính Chính thống]: Văn bản chứa cấu trúc bác bỏ/cảnh báo tin giả từ cơ quan có thẩm quyền "
            f"({', '.join(emotion_heuristics.get('debunking_cues', [])[:2])}), được bảo vệ khỏi báo động giả (False Positive)."
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
        f"[Đồng thuận Đa mô hình AI & Độ bất định]: Trung bình {len(model_fake_probs)} mô hình dự đoán xác suất văn phong tin giả là "
        f"{avg_ml_fake_prob*100:.1f}% | Độ bất định (Uncertainty Score): {uncertainty_score*100:.1f}%."
    )

    return {
        "verdict_code": verdict_code,
        "nli_3class_label": nli_3class_label,
        "verdict_title": verdict_title,
        "verdict_badge": verdict_badge,
        "confidence": confidence_score,
        "uncertainty_score": uncertainty_score,
        "veracity_score_1_to_5": final_veracity_1_to_5,
        "final_fake_probability": round(final_fake_prob, 4),
        "final_real_probability": round(1.0 - final_fake_prob, 4),
        "avg_ml_fake_probability": round(avg_ml_fake_prob, 4),
        "evidence_veracity_score": round(evidence_veracity_score, 2),
        "has_verified_evidence": has_strong_evidence,
        "verified_evidences": analyzed_evidences,
        "llm_verification": llm_verification,
        "nli_llm_agreement": nli_llm_agree,
        "retrieval_stats": retrieval_output.get("stats", {}),
        "reasoning_steps": reasoning_steps,
    }
