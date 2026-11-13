import os
import re
import json
import hashlib
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_CACHE_PATH = os.path.join(BASE_DIR, "models", "llm_cache.json")
_LLM_CACHE: Optional[Dict[str, Any]] = None
_NLI_BUNDLE: Optional[Dict[str, Any]] = None
_NLI_LOADED: bool = False


def _get_nli_bundle() -> Optional[Dict[str, Any]]:
    global _NLI_BUNDLE, _NLI_LOADED
    if _NLI_LOADED:
        return _NLI_BUNDLE
    _NLI_LOADED = True
    try:
        import joblib
        nli_path = os.path.join(BASE_DIR, "models", "nli_verifier_clf.pkl")
        if os.path.exists(nli_path):
            _NLI_BUNDLE = joblib.load(nli_path)
    except Exception:
        _NLI_BUNDLE = None
    return _NLI_BUNDLE


def _get_llm_cache() -> Dict[str, Any]:
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE
    if os.path.exists(LLM_CACHE_PATH):
        try:
            with open(LLM_CACHE_PATH, "r", encoding="utf-8") as f:
                _LLM_CACHE = json.load(f)
                return _LLM_CACHE
        except Exception:
            pass
    _LLM_CACHE = {}
    return _LLM_CACHE


def _save_llm_cache() -> None:
    if _LLM_CACHE is None:
        return
    try:
        os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
        with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_LLM_CACHE, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

REFUTATION_CUES_VN = [
    "bác bỏ", "sai sự thật", "tin giả", "bịa đặt", "thất thiệt", "không có chuyện",
    "đính chính", "tung tin đồn", "tuyệt đối không"
]

CONFIRMATION_CUES_VN = [
    "chính thức", "đã ký quyết định", "ban hành quyết định", "có hiệu lực",
    "xác nhận", "công bố", "theo nghị định", "theo thông tư", "thăng cấp bậc hàm",
    "triển khai", "khuyến cáo", "đưa vào khai thác", "thông xe", "vận hành thương mại",
    "phát đi cảnh báo chính thức"
]

UNCERTAINTY_CUES_VN = [
    "chưa đủ bằng chứng", "còn phụ thuộc", "đang lấy ý kiến", "đang được nghiên cứu",
    "chưa có quyết định", "dự kiến", "thử nghiệm", "tác động cụ thể phụ thuộc",
    "đang được hoàn thiện", "chưa có quy định"
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
    # Với tin RSS trực tuyến (LIVE_WEB_NEWS), chỉ lấy phần tiêu đề thực tế để tránh lẫn ngày xuất bản (vd: 12 Nov 2026) vào số liệu nội dung
    if evidence.get("tier") == "LIVE_WEB_NEWS" and evidence.get("title"):
        ev_content_only = str(evidence.get("title", "")).lower()
    else:
        ev_content_only = f"{evidence.get('title', '')} {evidence.get('evidence_text', '')}".lower()

    ev_title_and_body = ev_content_only
    known_score = evidence.get("known_verdict_score")
    known_nli = evidence.get("nli_label")

    has_refutation_in_ev = any(kw in ev_title_and_body for kw in REFUTATION_CUES_VN)
    has_confirmation_in_ev = any(kw in ev_title_and_body for kw in CONFIRMATION_CUES_VN)
    has_uncertainty_in_ev = any(kw in ev_title_and_body for kw in UNCERTAINTY_CUES_VN)

    # Chuẩn hóa chữ số đếm cơ bản trong tiếng Việt (vd: "hai hiệu trưởng" <-> "2 hiệu trưởng")
    def _norm_vn_nums(txt: str) -> str:
        num_words = {"một": "1", "hai": "2", "ba": "3", "bốn": "4", "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10"}
        out = txt
        for w, d in num_words.items():
            out = re.sub(rf"\b{w}\b", d, out)
        return out

    # Đối chiếu số liệu định lượng giữa Claim và Bằng chứng (Quantitative Consistency Check)
    c_nums = set(re.findall(r"\d+(?:[.,]\d+)?", _norm_vn_nums(claim_text.lower())))
    e_nums = set(re.findall(r"\d+(?:[.,]\d+)?", _norm_vn_nums(ev_title_and_body)))
    has_number_conflict = bool(c_nums and e_nums and not c_nums.intersection(e_nums))

    # Kiểm tra độ chồng lấp từ vựng thực tế giữa Claim và Evidence
    c_words = set(re.findall(r"\w{2,}", _norm_vn_nums(claim_text.lower())))
    e_words = set(re.findall(r"\w{2,}", _norm_vn_nums(ev_title_and_body)))
    lexical_overlap = len(c_words.intersection(e_words)) / max(len(c_words), 1) if c_words else 0.0

    # Chỉ tin trực tiếp known_score khi bằng chứng thuộc Kho chuẩn (LOCAL_KNOWLEDGE_BASE) và khớp cao (>= 0.72);
    # với HISTORICAL_DATASET (ReINTEL) hoặc độ khớp thấp, bắt buộc suy luận từ nội dung văn bản (tránh nhầm "tin thật" = "ủng hộ claim").
    use_known_shortcut = (
        known_score is not None
        and evidence.get("tier") != "HISTORICAL_DATASET"
        and float(evidence.get("similarity", 0.0)) >= 0.72
    )

    # Trường hợp 1: Bằng chứng đến từ Kho Kiểm chứng chuẩn đã gán nhãn và khớp cao
    if use_known_shortcut:
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
        # Trường hợp 2: Suy luận NLI trực tiếp từ nội dung bằng chứng (Strict Evaluation / Live Web News RSS)
        # 2A. Ưu tiên kiểm tra xung đột số liệu trực tiếp hoặc tín hiệu bác bỏ rõ ràng
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
                explanation = f"Bằng chứng chính thống ({evidence.get('source')}) đã bác bỏ hoặc cảnh báo thông tin này là sai sự thật/lừa đảo."
        elif has_uncertainty_in_ev:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = f"Nguồn tin ({evidence.get('source')}) cho biết vấn đề đang trong quá trình nghiên cứu/lấy ý kiến hoặc phụ thuộc điều kiện cụ thể, chưa đủ căn cứ kết luận (NEI)."
        elif has_number_conflict and not claim_is_debunking:
            stance = "NEI"
            nli_label = "NEI"
            veracity_score = 3
            explanation = f"Bằng chứng trên {evidence.get('source')} đề cập cùng chủ đề nhưng số liệu/mốc thời gian ({', '.join(sorted(e_nums)[:3])}) không khớp với mệnh đề ({', '.join(sorted(c_nums)[:3])})."
        elif evidence.get("tier") == "LIVE_WEB_NEWS" and float(evidence.get("similarity", 0)) >= 0.80 and lexical_overlap >= 0.50:
            stance = "SUPPORTS"
            nli_label = "Supported"
            veracity_score = 5
            explanation = f"Báo chí chính thống ({evidence.get('source')}) đăng bài viết có tiêu đề khớp trực tiếp ({lexical_overlap*100:.0f}% từ khóa) xác nhận sự kiện này."
        else:
            # 2B. Sử dụng Mô hình 3-Class NLI Verifier đã huấn luyện nếu có
            nli_bundle = _get_nli_bundle()
            used_ml_nli = False
            if nli_bundle is not None:
                try:
                    from src.train_nli_verifier import extract_nli_pair_features
                    from src.preprocess import normalize_vietnamese_unicode
                    ev_full = f"{evidence.get('title', '')}. {evidence.get('evidence_text', '')}"
                    combined_str = normalize_vietnamese_unicode(f"CLAIM: {claim_text} || EVIDENCE: {ev_full}".lower())
                    x_tf = nli_bundle["tfidf"].transform([combined_str]).toarray()
                    x_pair = extract_nli_pair_features([(claim_text, ev_full)])
                    x_in = np.hstack([x_tf * 0.45, x_pair * 1.85])
                    probs = nli_bundle["clf"].predict_proba(x_in)[0]
                    pred_idx = int(np.argmax(probs))
                    pred_lbl = nli_bundle["labels"][pred_idx]
                    if pred_lbl == "Supported":
                        stance, nli_label, veracity_score = "SUPPORTS", "Supported", 5
                        explanation = f"Mô hình NLI xác nhận mệnh đề khớp với bằng chứng trên {evidence.get('source')} (P={probs[pred_idx]:.2f})."
                    elif pred_lbl == "Refuted":
                        stance, nli_label, veracity_score = "REFUTES", "Refuted", 1
                        explanation = f"Mô hình NLI phát hiện mâu thuẫn với bằng chứng trên {evidence.get('source')} (P={probs[pred_idx]:.2f})."
                    else:
                        stance, nli_label, veracity_score = "NEI", "NEI", 3
                        explanation = f"Mô hình NLI đánh giá bằng chứng trên {evidence.get('source')} chưa đủ căn cứ kết luận (P={probs[pred_idx]:.2f})."
                    used_ml_nli = True
                except Exception:
                    used_ml_nli = False

            if not used_ml_nli:
                if (has_confirmation_in_ev or evidence.get("similarity", 0) >= 0.65) and lexical_overlap >= 0.28:
                    stance = "SUPPORTS"
                    nli_label = "Supported"
                    veracity_score = 4 if not has_confirmation_in_ev else 5
                    explanation = f"Bằng chứng chính thống ({evidence.get('source')}) đưa tin xác nhận sự kiện/quyết định và khớp số liệu."
                else:
                    stance = "NEI"
                    nli_label = "NEI"
                    veracity_score = 3
                    explanation = f"Bài báo trên {evidence.get('source')} đề cập một số từ khóa liên quan nhưng nội dung không đủ căn cứ khẳng định trực tiếp mệnh đề."

    enriched = dict(evidence)
    enriched["stance"] = stance
    enriched["nli_label"] = nli_label
    enriched["veracity_score_1_to_5"] = veracity_score
    enriched["stance_explanation"] = explanation
    enriched["citation_tag"] = f"[{evidence.get('id', 'EV')} | {evidence.get('source', 'Nguồn')} ({evidence.get('date', 'N/A')})]"
    return enriched


def _call_gemini_few_shot_api(
    claim_text: str,
    analyzed_evidences: List[Dict[str, Any]],
    few_shot_prompt_header: str
) -> Optional[Dict[str, Any]]:
    """
    Gọi trực tiếp Google Gemini API (3-Shot Evidence-Grounded Verification) với Disk Cache.
    Trả về None nếu không có API key hoặc lỗi mạng để hệ thống chuyển sang fallback an toàn.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or not analyzed_evidences:
        return None

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    ev_blocks = []
    for idx, ev in enumerate(analyzed_evidences[:3], 1):
        tag = ev.get("citation_tag", f"[{ev.get('id', f'EV-{idx}')}]")
        title = ev.get("title", "")
        body = ev.get("evidence_text", "")
        ev_blocks.append(f"Bằng chứng #{idx} {tag}: {title}. {body}")
    ev_context = "\n".join(ev_blocks)

    cache_key_raw = f"{model_name}||{claim_text.strip()}||{ev_context.strip()}"
    cache_key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()[:24]
    cache = _get_llm_cache()
    if cache_key in cache:
        cached_val = dict(cache[cache_key])
        cached_val["from_cache"] = True
        return cached_val

    full_prompt = (
        f"{few_shot_prompt_header}\n\n"
        f"--- CA KIỂM CHỨNG THỰC TẾ ---\n"
        f"Mệnh đề cần kiểm chứng (Claim): \"{claim_text}\"\n"
        f"Các Bằng chứng truy xuất được:\n{ev_context}\n\n"
        f"YÊU CẦU ĐẦU RA: Chỉ trả về duy nhất một đối tượng JSON hợp lệ (không bọc markdown) theo cấu trúc:\n"
        f'{{"label": "Supported" | "Refuted" | "NEI", "confidence": 0.92, "citations": ["[ID | Nguồn (Ngày)]"], '
        f'"explanation": "Câu giải thích ngắn gọn bằng tiếng Việt có trích dẫn mã bằng chứng và nêu rõ lý do."}}'
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 600,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp_data = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as he:
                if he.code == 429 and attempt == 0:
                    import time as _t
                    _t.sleep(4.2)
                    continue
                raise
        if not resp_data:
            return None

        candidates = resp_data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts).strip()
        # Làm sạch nếu có khối ```json
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
        parsed = json.loads(raw_text)

        lbl_raw = str(parsed.get("label", "NEI")).strip()
        if lbl_raw.lower().startswith("sup"):
            llm_label, llm_stance = "Supported", "SUPPORTS"
        elif lbl_raw.lower().startswith("ref") or lbl_raw.lower() == "false":
            llm_label, llm_stance = "Refuted", "REFUTES"
        else:
            llm_label, llm_stance = "NEI", "NEI"

        conf = float(np.clip(float(parsed.get("confidence", 0.85)), 0.50, 0.99))
        exp = str(parsed.get("explanation", "")).strip()
        cits = parsed.get("citations") or [analyzed_evidences[0].get("citation_tag", "")]

        result = {
            "llm_label": llm_label,
            "llm_stance": llm_stance,
            "llm_confidence": round(conf, 3),
            "citations_used": cits[:2],
            "grounded_explanation": exp,
            "model_provider": f"Google Gemini ({model_name})",
            "is_live_llm": True,
            "from_cache": False,
        }
        cache[cache_key] = result
        _save_llm_cache()
        return result
    except Exception:
        return None


def run_llm_few_shot_verification(
    claim_text: str,
    analyzed_evidences: List[Dict[str, Any]],
    is_debunking: bool = False,
    use_live_api: bool = True
) -> Dict[str, Any]:
    """
    Yêu cầu AI/ML trong Danh_sach_de_tai.xlsx (Đề tài số 8):
    "So sánh NLI/Transformer verification với LLM few-shot / evidence-grounded verification".
    Thực hiện suy luận Few-shot (3-shot In-Context Learning) qua Google Gemini API
    (có cơ chế Cache và Fallback nội bộ khi không có kết nối mạng).
    """
    few_shot_prompt_preview = (
        "SYSTEM: Bạn là Hệ thống Kiểm chứng Thông tin Tiếng Việt dựa trên Bằng chứng (Evidence-Grounded Fact-Checker).\n"
        "NHIỆM VỤ: Dựa duy nhất vào các Bằng chứng (Evidence) được cung cấp, phân loại Mệnh đề (Claim) vào 1 trong 3 nhãn:\n"
        "  - Supported: Bằng chứng xác nhận mệnh đề là đúng sự thật.\n"
        "  - Refuted: Bằng chứng bác bỏ trực tiếp mệnh đề, khẳng định là tin giả/sai sự thật/lừa đảo.\n"
        "  - NEI: Bằng chứng chỉ mới ở mức đang nghiên cứu, đang lấy ý kiến, thử nghiệm, chưa có quyết định ấn định, hoặc chưa đủ căn cứ khẳng định/bác bỏ.\n"
        "--- 3-SHOT EXAMPLES ---\n"
        "Ví dụ 1: Claim: 'NHNN phát hành tiền 1 triệu đồng' + Evidence [KB-001]: 'NHNN khẳng định thông tin tiền 1 triệu đồng là bịa đặt' -> Nhãn: Refuted.\n"
        "Ví dụ 2: Claim: 'Tăng lương cơ sở lên 2,34 triệu từ 1/7/2024' + Evidence [KB-006]: 'Nghị định 73/2024 quy định lương cơ sở 2.340.000đ' -> Nhãn: Supported.\n"
        "Ví dụ 3: Claim: 'Các trường ĐH sẽ dành 15% chỉ tiêu riêng cho chứng chỉ quốc tế' + Evidence [VNFC-009]: 'Một số trường đang lấy ý kiến đề xuất cộng điểm, chưa có quyết định ấn định tỷ lệ 15%' -> Nhãn: NEI."
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
            "model_provider": "Evidence-Grounded Guardrail (No Evidence >= 0.50)",
            "is_live_llm": False,
            "prompt_template": few_shot_prompt_preview
        }

    # 1. Thử gọi trực tiếp Google Gemini 3-Shot API (hoặc lấy từ Cache đã xác thực)
    if use_live_api:
        gemini_res = _call_gemini_few_shot_api(claim_text, analyzed_evidences, few_shot_prompt_preview)
        if gemini_res is not None:
            gemini_res["prompt_template"] = few_shot_prompt_preview
            return gemini_res

    top_ev = analyzed_evidences[0]
    ev_text_lower = f"{top_ev.get('title', '')} {top_ev.get('evidence_text', '')}".lower()
    citation = top_ev.get("citation_tag", f"[{top_ev.get('id', 'EV')}]")

    # 2. Fallback: Suy luận Few-shot dựa trên đối chiếu mệnh đề và nội dung bằng chứng
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
        "model_provider": "Local Rule-Grounded Fallback",
        "is_live_llm": False,
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
        decisive_evs = [ev for ev in analyzed_evidences if ev.get("stance") in ("SUPPORTS", "REFUTES")]
        top_ev = analyzed_evidences[0]
        if decisive_evs and (top_ev.get("stance") in ("SUPPORTS", "REFUTES") or float(decisive_evs[0].get("similarity", 0)) >= 0.75):
            # Các bằng chứng đã xác định rõ SUPPORTS/REFUTES quyết định điểm thực chứng;
            # không để các tin phụ dạng NEI kéo loãng điểm của bằng chứng trực tiếp.
            weights = [max(0.1, float(ev.get("similarity", 0.5))) ** 2 for ev in decisive_evs]
            scores_1_5 = [float(ev.get("veracity_score_1_to_5", 3)) for ev in decisive_evs]
            evidence_veracity_score = float(np.average(scores_1_5, weights=weights))
            stance_std = float(np.std(scores_1_5)) if len(scores_1_5) > 1 else 0.0
        else:
            weights = [max(0.1, float(ev.get("similarity", 0.5))) for ev in analyzed_evidences]
            scores_1_5 = [float(ev.get("veracity_score_1_to_5", 3)) for ev in analyzed_evidences]
            evidence_veracity_score = float(np.average(scores_1_5, weights=weights))
            stance_std = float(np.std(scores_1_5)) if len(scores_1_5) > 1 else 0.0

        evidence_fake_prob = (5.0 - evidence_veracity_score) / 4.0
        # Độ bất định (Uncertainty) dựa trên độ lệch chuẩn giữa các bằng chứng và khoảng cách tới điểm trung lập 3.0
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
        top_sim = float(analyzed_evidences[0].get("similarity", 0.5))
        llm_lbl = llm_verification.get("llm_label", "NEI")
        llm_conf = float(llm_verification.get("llm_confidence", 0.55))
        is_live_llm = bool(llm_verification.get("is_live_llm", False))

        if llm_lbl == "Refuted":
            llm_fake_prob = llm_conf
        elif llm_lbl == "Supported":
            llm_fake_prob = 1.0 - llm_conf
        else:
            llm_fake_prob = 0.50

        nli_top_lbl = analyzed_evidences[0].get("nli_label", "NEI")
        has_nli_llm_conflict = (
            is_live_llm
            and nli_top_lbl in ("Supported", "Refuted")
            and llm_lbl in ("Supported", "Refuted")
            and nli_top_lbl != llm_lbl
        )

        if has_nli_llm_conflict:
            # Khi NLI nội bộ và Live LLM đối lập trực tiếp -> tăng độ bất định & cân bằng phán quyết
            uncertainty_score = round(max(uncertainty_score, 0.72), 3)
            if top_sim < 0.80:
                combined_fake_prob = (
                    0.25 * evidence_fake_prob
                    + 0.55 * llm_fake_prob
                    + 0.20 * avg_ml_fake_prob
                )
            else:
                combined_fake_prob = 0.50 + 0.15 * (llm_fake_prob - 0.50)
        elif top_stance == "NEI":
            if is_live_llm and llm_lbl in ("Supported", "Refuted"):
                combined_fake_prob = 0.50 + 0.20 * (llm_fake_prob - 0.50) + 0.08 * (avg_ml_fake_prob - 0.50)
            else:
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
