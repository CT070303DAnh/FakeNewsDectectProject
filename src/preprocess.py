import os
import re
import unicodedata
from typing import Dict, List, Any
import urllib.request
from urllib.parse import urlparse
import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    joblib = None
    HAS_JOBLIB = False

_CW_MODEL_BUNDLE = None
_CW_MODEL_LOADED = False

try:
    from underthesea import word_tokenize, ner
except ImportError:
    def word_tokenize(text, format=None):
        return text.split() if format != "text" else " ".join(text.split())
    def ner(text):
        return []

# Từ điển chuẩn hóa viết tắt / teencode phổ biến trên mạng xã hội Việt Nam
TEENCODE_MAP = {
    "ko": "không",
    "k": "không",
    "khg": "không",
    "đc": "được",
    "dc": "được",
    "vs": "với",
    "mn": "mọi người",
    "ng": "người",
    "nch": "nói chuyện",
    "trc": "trước",
    "bh": "bây giờ",
    "nhn": "ngân hàng nhà nước",
    "bca": "bộ công an",
    "byt": "bộ y tế",
    "bđs": "bất động sản",
    "ctv": "cộng tác viên",
    "stk": "số tài khoản",
    "cccd": "căn cước công dân",
}

STOPWORDS_QUERY = {
    "là", "của", "và", "các", "những", "này", "đó", "được", "trong", "cho",
    "với", "một", "có", "không", "đã", "đang", "sẽ", "về", "trên", "tại",
    "theo", "từ", "đến", "như", "khi", "mà", "thì", "vào", "ra", "lên",
    "xuống", "rất", "quá", "lắm", "nhé", "nha", "ạ", "ơi", "hãy", "cùng"
}

# Tín hiệu đặc trưng cho bài toán Phân loại Check-worthiness (Check-worthiness Classification)
FACTUAL_CLAIM_INDICATORS = [
    "ban hành", "quy định", "nghị định", "quyết định", "thông tư", "công bố",
    "xác nhận", "bác bỏ", "cảnh báo", "phát hành", "triển khai", "chính thức",
    "tăng", "giảm", "mức lương", "học phí", "lãi suất", "tịch thu", "khóa",
    "chữa khỏi", "ung thư", "vắc-xin", "tiêm chủng", "sập", "vỡ đập", "phong tỏa",
    "ngân hàng", "bộ y tế", "bộ công an", "bộ giáo dục", "chính phủ", "quốc hội",
    "cục hàng không", "tập đoàn điện lực", "evn", "vneid", "bảo hiểm xã hội",
    "triệu đồng", "tỷ đồng", "nghìn tỷ", "%", "km", "tháng", "năm"
]

SUBJECTIVE_CHITCHAT_INDICATORS = [
    "chúc cả nhà", "chúc mừng sinh nhật", "theo ý kiến cá nhân", "theo gu thẩm mỹ",
    "mình cảm thấy", "ước gì", "điểm danh cái nào", "thả tim", "hỏi quanh khu vực",
    "cảm ơn tất cả anh chị em", "bấm vào trang cá nhân của shop", "vui vẻ nhé",
    "haha", "hihi", "buồn và mệt mỏi"
]


def normalize_vietnamese_unicode(text: str) -> str:
    """Chuẩn hóa bảng mã Unicode tiếng Việt về chuẩn NFC."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def _load_cw_model():
    global _CW_MODEL_BUNDLE, _CW_MODEL_LOADED
    if _CW_MODEL_LOADED:
        return _CW_MODEL_BUNDLE
    _CW_MODEL_LOADED = True
    if not HAS_JOBLIB:
        return None
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "check_worthiness_clf.pkl")
    if os.path.exists(model_path):
        try:
            _CW_MODEL_BUNDLE = joblib.load(model_path)
        except Exception:
            _CW_MODEL_BUNDLE = None
    return _CW_MODEL_BUNDLE


def _extract_single_structural_features(norm_s: str) -> np.ndarray:
    lower = norm_s.lower()
    words = lower.split()
    n_words = max(len(words), 1)
    factual_hits = sum(1 for kw in FACTUAL_CLAIM_INDICATORS if kw in lower)
    subj_hits = sum(1 for kw in SUBJECTIVE_CHITCHAT_INDICATORS if kw in lower)
    personal_pronouns = sum(
        1 for w in ["mình", "tôi", "em", "shop", "cả nhà", "mọi người", "các bác", "anh em", "nhé", "nha", "ạ", "haha", "hihi", "ơi"]
        if re.search(rf"\b{re.escape(w)}\b", lower)
    )
    official_cues = sum(
        1 for w in ["theo", "quyết định", "nghị định", "thông tư", "bộ", "cục", "ngân hàng", "chính phủ", "công bố", "xác nhận", "bác bỏ", "cảnh báo", "triển khai", "ban hành", "quy định", "tuyển sinh", "vắc-xin", "lãi suất", "tăng", "giảm"]
        if w in lower
    )
    num_count = len(re.findall(r"\d+", lower))
    has_date_pattern = 1.0 if re.search(r"\b(\d{1,2}/\d{1,2}|20\d{2}|tháng\s+\d+|năm\s+20\d{2})\b", lower) else 0.0
    has_unit_pattern = 1.0 if re.search(r"\b(triệu|tỷ|nghìn|%|\bđồng|usd|km|tuổi|ca\b|tỉnh|thành phố)\b", lower) else 0.0
    exclam_question = norm_s.count("!") + norm_s.count("?")
    return np.asarray([[
        min(1.0, n_words / 35.0),
        min(1.0, factual_hits / 3.0),
        min(1.0, subj_hits / 2.0),
        min(1.0, personal_pronouns / 2.0),
        min(1.0, official_cues / 3.0),
        min(1.0, num_count / 3.0),
        has_date_pattern,
        has_unit_pattern,
        min(1.0, exclam_question / 3.0),
    ]], dtype=np.float32)


def predict_check_worthiness(sentence: str) -> Dict[str, Any]:
    """
    Node 3A: Mô hình phân loại Check-worthiness (Hybrid ML Classifier: TF-IDF + Linguistic Features
    kết hợp Heuristic Fallback).
    Xác định một câu có phải là mệnh đề thực tế cần kiểm chứng (Check-worthy Factual Claim)
    hay chỉ là câu cảm thán, ý kiến cá nhân, lời chào hỏi trên mạng xã hội.
    """
    norm_s = normalize_vietnamese_unicode(sentence).strip()
    lower_s = norm_s.lower()

    if len(norm_s) < 12:
        return {
            "sentence": norm_s,
            "is_check_worthy": False,
            "check_worthiness_score": 0.10,
            "claim_type": "Too Short / Fragment",
            "model_used": "LENGTH_FILTER"
        }

    # Ưu tiên sử dụng mô hình ML Check-worthiness đã huấn luyện nếu có
    bundle = _load_cw_model()
    if bundle is not None and "tfidf" in bundle and "clf" in bundle:
        try:
            x_tfidf = bundle["tfidf"].transform([lower_s]).toarray()
            x_struct = _extract_single_structural_features(norm_s)
            x_full = np.hstack([x_tfidf * 0.65, x_struct * 1.35])
            prob = float(bundle["clf"].predict_proba(x_full)[0][1])
            score = max(0.02, min(0.99, round(prob, 3)))
            is_worthy = score >= 0.50
            return {
                "sentence": norm_s,
                "is_check_worthy": is_worthy,
                "check_worthiness_score": score,
                "claim_type": "Factual Claim (Đáng kiểm chứng)" if is_worthy else "Opinion / Social Chitchat (Không cần kiểm chứng)",
                "model_used": "ML_LOGISTIC_TFIDF_STRUCTURAL"
            }
        except Exception:
            pass

    # Fallback: Đếm tín hiệu thực tế (số liệu, ngày tháng, tổ chức, động từ khẳng định/chính sách)
    factual_hits = sum(1 for kw in FACTUAL_CLAIM_INDICATORS if kw in lower_s)
    has_numbers = bool(re.search(r"\d+", lower_s))
    has_capitals = bool(re.search(r"[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]", norm_s[1:]))
    subjective_hits = sum(1 for kw in SUBJECTIVE_CHITCHAT_INDICATORS if kw in lower_s)

    score = 0.45
    score += min(0.42, factual_hits * 0.14)
    if has_numbers:
        score += 0.15
    if has_capitals:
        score += 0.08
    score -= min(0.65, subjective_hits * 0.35)

    score = max(0.02, min(0.99, round(score, 3)))
    is_worthy = score >= 0.50

    return {
        "sentence": norm_s,
        "is_check_worthy": is_worthy,
        "check_worthiness_score": score,
        "claim_type": "Factual Claim (Đáng kiểm chứng)" if is_worthy else "Opinion / Social Chitchat (Không cần kiểm chứng)",
        "model_used": "HEURISTIC_FALLBACK"
    }


def extract_text_from_url_if_needed(raw_input: str) -> Dict[str, Any]:
    """
    Node 1: Kiểm tra nếu đầu vào là một URL bài báo thì tự động tải và trích xuất nội dung.
    Canonicalization theo Mục S2.2.1 của bài báo Vosoughi et al. (2018).
    """
    stripped = raw_input.strip()
    parsed = urlparse(stripped)
    if parsed.scheme in ("http", "https") and parsed.netloc and " " not in stripped:
        try:
            req = urllib.request.Request(
                stripped,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.IGNORECASE | re.DOTALL)
            clean_paras = []
            for p in paragraphs:
                p_txt = re.sub(r"<[^>]+>", " ", p)
                p_txt = re.sub(r"\s+", " ", p_txt).strip()
                if len(p_txt) > 40:
                    clean_paras.append(p_txt)
            body = " ".join(clean_paras[:8])
            extracted_text = f"{title}. {body}".strip()
            if len(extracted_text) > 30:
                return {
                    "is_url": True,
                    "source_url": stripped,
                    "domain": parsed.netloc,
                    "text": extracted_text,
                    "title": title
                }
        except Exception:
            pass
    return {
        "is_url": False,
        "source_url": None,
        "domain": None,
        "text": raw_input,
        "title": None
    }


def clean_text(text: str) -> str:
    """
    Node 2: Tiền xử lý văn bản tiếng Việt (Lowercase, Chuẩn hóa NFC, Teencode, Tách từ ghép).
    Bảo toàn dấu ! và ? để đo cảm xúc/giật gân.
    """
    if not isinstance(text, str):
        return ""

    text = normalize_vietnamese_unicode(text)
    text = text.lower()

    # Loại bỏ URL khỏi văn bản phân loại nhưng giữ nội dung chữ
    text = re.sub(r"https?://\S+", " ", text)

    # Giữ lại chữ cái, số, khoảng trắng và dấu ! ?
    text = re.sub(r"[^\w\s!?]", " ", text)
    text = re.sub(r"([!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Chuẩn hóa teencode
    words = [TEENCODE_MAP.get(w, w) for w in text.split()]
    text = " ".join(words)

    try:
        tokens = word_tokenize(text, format="text")
    except Exception:
        tokens = " ".join(text.split())

    return tokens


def extract_claims_and_queries(raw_text: str) -> Dict[str, Any]:
    """
    Node 3: Phát hiện Mệnh đề đáng kiểm chứng (Claim Detection & Check-worthiness),
    Chuẩn hóa Claim (Claim Normalization) & Trích xuất Thực thể/Thời gian (NER & Temporal Cues)
    để sinh truy vấn tìm kiếm bằng chứng (Search Queries).
    """
    text = normalize_vietnamese_unicode(raw_text).strip()
    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?;\n])\s+", text)
        if len(s.strip()) > 12
    ]
    if not raw_sentences and text:
        raw_sentences = [text]

    social_fluff_patterns = re.compile(
        r"^(mọi người ơi|share mạnh|chia sẻ ngay|bấm vào link|comment ngay|inbox ngay)",
        re.IGNORECASE
    )

    sentence_evaluations: List[Dict[str, Any]] = []
    factual_claims: List[str] = []

    for sent in raw_sentences:
        cleaned_sent = re.sub(r"\s+", " ", sent).strip()
        # Loại bỏ cụm cảm thán rác ở đầu câu (Claim Normalization)
        normalized_claim = re.sub(
            r"^(tin chấn động|thông báo khẩn|tin khẩn|khẩn cấp|chấn động|nóng)\s*[:!-]*\s*",
            "",
            cleaned_sent,
            flags=re.IGNORECASE
        ).strip()
        cw = predict_check_worthiness(normalized_claim)
        sentence_evaluations.append(cw)
        if cw["is_check_worthy"] and not social_fluff_patterns.search(cleaned_sent):
            factual_claims.append(normalized_claim)

    if not factual_claims:
        factual_claims = [re.sub(r"\s+", " ", s).strip() for s in raw_sentences[:2]]

    # Chọn mệnh đề có điểm Check-worthiness cao nhất làm Primary Claim
    worthy_sorted = sorted(sentence_evaluations, key=lambda x: x["check_worthiness_score"], reverse=True)
    primary_claim = worthy_sorted[0]["sentence"] if worthy_sorted else factual_claims[0]
    primary_cw_score = worthy_sorted[0]["check_worthiness_score"] if worthy_sorted else 0.75

    # Trích xuất thực thể (Tên riêng, Tổ chức, Con số, Ngày tháng)
    entities: List[str] = []
    try:
        ner_tags = ner(primary_claim)
        for item in ner_tags:
            if len(item) >= 4 and item[3] != "O":
                ent = item[0].strip()
                if len(ent) > 1 and ent not in entities:
                    entities.append(ent)
    except Exception:
        pass

    capitalized_phrases = re.findall(
        r"\b(?:[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+\s*){1,4}",
        primary_claim
    )
    for phrase in capitalized_phrases:
        p = phrase.strip()
        if len(p) > 3 and p.lower() not in STOPWORDS_QUERY and p not in entities:
            entities.append(p)

    numbers_and_dates = re.findall(
        r"\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}|\d+(?:[.,]\d+)?\s*(?:%|triệu|tỷ|nghìn|ngàn|đồng|usd|ngày|tháng|năm|ca|người|km|tuổi|tiếng)?)\b",
        text,
        re.IGNORECASE
    )
    numbers_clean = []
    for n in numbers_and_dates:
        n_str = n.strip()
        if len(n_str) >= 2 and n_str not in numbers_clean:
            numbers_clean.append(n_str)

    # Tạo câu truy vấn tìm kiếm bằng chứng (Search Queries)
    words = re.sub(r"[^\w\s]", " ", primary_claim).split()
    keywords = [w for w in words if w.lower() not in STOPWORDS_QUERY and len(w) > 1]
    query_primary = " ".join(keywords[:14])

    queries = []
    if query_primary:
        queries.append(query_primary)
    if entities:
        ent_query = " ".join(entities[:4] + numbers_clean[:2])
        if ent_query and ent_query not in queries:
            queries.append(ent_query)

    return {
        "primary_claim": primary_claim,
        "check_worthiness_score": primary_cw_score,
        "sentence_check_worthiness": sentence_evaluations[:5],
        "all_claims": factual_claims[:4],
        "entities": entities[:8],
        "numbers_dates": numbers_clean[:6],
        "search_queries": queries[:2] if queries else [text[:100]]
    }


if __name__ == "__main__":
    sample = "Tin khẩn: Ngân hàng Nhà nước chuẩn bị phát hành tờ tiền mệnh giá 1 triệu đồng từ tháng sau! Share ngay!"
    print("Cleaned:", clean_text(sample))
    print("Claims & Queries:", extract_claims_and_queries(sample))
