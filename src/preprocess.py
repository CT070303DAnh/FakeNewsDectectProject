import re
import unicodedata
from typing import Dict, List, Any
import urllib.request
from urllib.parse import urlparse

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

FACTUAL_VERBS_AND_CUES = [
    "ban hành", "quyết định", "thông báo", "xác nhận", "công bố", "phát hành",
    "tăng", "giảm", "xử phạt", "khởi tố", "bắt giữ", "thu hồi", "cấm",
    "chữa khỏi", "phát hiện", "vỡ đập", "phong tỏa", "sập", "tai nạn",
    "nghị định", "thông tư", "luật", "bộ y tế", "bộ công an", "chính phủ",
    "ngân hàng", "ubnd", "tỷ đồng", "triệu đồng", "phần trăm", "%"
]

SUBJECTIVE_OR_FLUFF_CUES = [
    "mọi người ơi", "share mạnh", "chia sẻ ngay", "bấm vào link", "comment ngay",
    "inbox ngay", "buồn quá", "vui quá", "theo cảm nhận của mình", "chúc mọi người",
    "ai cần liên hệ", "tuyển sỉ", "xả kho"
]


def normalize_vietnamese_unicode(text: str) -> str:
    """Chuẩn hóa bảng mã Unicode tiếng Việt về chuẩn NFC."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


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


def classify_check_worthiness(sentence: str) -> Dict[str, Any]:
    """
    Node 3A: Phân loại Check-worthiness (Xác định câu có chứa mệnh đề thực tế đáng kiểm chứng hay không).
    Đánh giá dựa trên mật độ thực thể định danh (NER), số liệu/mốc thời gian, động từ khẳng định sự kiện
    và trừ điểm với các câu cảm thán/kêu gọi tương tác MXH.
    """
    s_clean = normalize_vietnamese_unicode(sentence).strip()
    s_lower = s_clean.lower()

    if len(s_clean) < 18:
        return {"sentence": s_clean, "score": 0.15, "is_check_worthy": False, "reason": "Câu quá ngắn"}

    score = 0.45
    reasons = []

    # 1. Có chứa con số, tỷ lệ %, tiền tệ hoặc ngày tháng
    num_matches = re.findall(
        r"\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d+(?:[.,]\d+)?\s*(?:%|triệu|tỷ|nghìn|đồng|usd|ngày|tháng|năm|ca|người)?)\b",
        s_lower
    )
    if num_matches:
        score += min(0.28, 0.12 * len(num_matches))
        reasons.append("Chứa số liệu/thời gian định lượng")

    # 2. Có chứa từ khóa sự kiện/chính sách/kiểm chứng
    factual_hits = [kw for kw in FACTUAL_VERBS_AND_CUES if kw in s_lower]
    if factual_hits:
        score += min(0.25, 0.09 * len(factual_hits))
        reasons.append(f"Chứa vị ngữ sự kiện ({', '.join(factual_hits[:2])})")

    # 3. Có chứa cụm từ viết hoa (Tên cơ quan, địa danh, nhân vật)
    cap_matches = re.findall(
        r"\b[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]+",
        s_clean[1:] if len(s_clean) > 1 else s_clean
    )
    if cap_matches:
        score += min(0.18, 0.06 * len(cap_matches))
        reasons.append("Chứa thực thể tên riêng/tổ chức")

    # 4. Giảm điểm nếu là câu kêu gọi tương tác hoặc cảm thán cá nhân
    fluff_hits = [kw for kw in SUBJECTIVE_OR_FLUFF_CUES if kw in s_lower]
    if fluff_hits:
        score -= 0.40
        reasons.append("Câu kêu gọi tương tác/cảm nghĩ cá nhân")

    final_score = round(max(0.05, min(0.99, score)), 3)
    is_worthy = final_score >= 0.50

    return {
        "sentence": s_clean,
        "score": final_score,
        "is_check_worthy": is_worthy,
        "reason": "; ".join(reasons) if reasons else "Mệnh đề trần thuật thông thường"
    }


def extract_claims_and_queries(raw_text: str, use_crf_ner: bool = False) -> Dict[str, Any]:
    """
    Node 3: Phát hiện Claim đáng kiểm chứng (Claim Detection / Check-worthiness),
    Chuẩn hóa Claim (Claim Normalization) & Trích xuất Thực thể (NER) để sinh Search Queries.
    """
    text = normalize_vietnamese_unicode(raw_text).strip()
    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?;\n])\s+", text)
        if len(s.strip()) > 12
    ]
    if not raw_sentences and text:
        raw_sentences = [text]

    # Phân loại Check-worthiness cho từng câu
    sentence_evaluations = [classify_check_worthiness(s) for s in raw_sentences]
    worthy_items = [item for item in sentence_evaluations if item["is_check_worthy"]]
    worthy_items.sort(key=lambda x: x["score"], reverse=True)

    factual_claims = [item["sentence"] for item in worthy_items]
    if not factual_claims:
        factual_claims = [raw_sentences[0]]

    # Chọn mệnh đề có điểm Check-worthiness cao nhất làm Primary Claim
    primary_claim = factual_claims[0] if factual_claims else text[:300]
    primary_check_worthiness = worthy_items[0]["score"] if worthy_items else sentence_evaluations[0]["score"]

    # Chuẩn hóa Claim (Claim Normalization: loại bỏ từ cảm thán đầu câu, chuẩn hóa khoảng trắng)
    normalized_claim = re.sub(
        r"^(?:tin khẩn|khẩn cấp|chấn động|nóng|cảnh báo|thông báo khẩn)\s*[:!-]*\s*",
        "",
        primary_claim,
        flags=re.IGNORECASE
    ).strip()

    # Trích xuất thực thể (Tên riêng, Tổ chức, Con số, Ngày tháng)
    entities: List[str] = []
    if use_crf_ner:
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
        r"\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d+(?:[.,]\d+)?\s*(?:%|triệu|tỷ|nghìn|ngàn|đồng|usd|ngày|tháng|năm|ca|người)?)\b",
        text,
        re.IGNORECASE
    )
    numbers_clean = []
    for n in numbers_and_dates:
        n_str = n.strip()
        if len(n_str) >= 2 and n_str not in numbers_clean:
            numbers_clean.append(n_str)

    # Trích xuất mốc năm (Temporal Year) nếu có trong claim để phục vụ Temporal Filtering ở Giai đoạn 2
    years_in_claim = re.findall(r"\b(20\d{2})\b", text)
    claim_temporal_year = int(years_in_claim[0]) if years_in_claim else None

    # Tạo câu truy vấn tìm kiếm bằng chứng (Search Queries)
    words = re.sub(r"[^\w\s]", " ", normalized_claim).split()
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
        "normalized_claim": normalized_claim,
        "check_worthiness_score": primary_check_worthiness,
        "sentence_check_worthiness": sentence_evaluations[:6],
        "all_claims": factual_claims[:4],
        "entities": entities[:8],
        "numbers_dates": numbers_clean[:6],
        "claim_temporal_year": claim_temporal_year,
        "search_queries": queries[:2] if queries else [text[:100]]
    }


if __name__ == "__main__":
    sample = "Mọi người ơi chia sẻ gấp!! Tin khẩn: Ngân hàng Nhà nước chuẩn bị phát hành tờ tiền mệnh giá 1 triệu đồng từ tháng 8/2024!"
    print("Cleaned:", clean_text(sample))
    print("Claims & Queries:", extract_claims_and_queries(sample))
