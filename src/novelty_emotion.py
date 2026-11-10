import re
import numpy as np
from typing import Dict, List, Any, Optional

# =====================================================================
# NODE 8: ĐO ĐỘ MỚI LẠ THÔNG TIN (INFORMATION NOVELTY - MỤC S7.2 SCIENCE 2018)
# =====================================================================

def _to_probability_distribution(vec: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    Chuyển vector biểu diễn ngữ nghĩa/chủ đề thành phân phối xác suất rời rạc P(i) > 0
    để tính KL-Divergence (Công thức S7, S8) và Bhattacharyya Distance (Công thức S9).
    """
    v = np.asarray(vec, dtype=np.float64).flatten()
    if v.size == 0:
        v = np.ones(100, dtype=np.float64)
    elif np.allclose(v, 0):
        v = np.ones_like(v, dtype=np.float64)
    # Chuẩn hóa Z-score trước khi đưa qua Softmax để tạo phân phối chủ đề sắc nét
    std = np.std(v)
    if std > 1e-8:
        v = (v - np.mean(v)) / std
    scaled = v / max(temperature, 1e-4)
    scaled = scaled - np.max(scaled)
    exp_v = np.exp(scaled)
    prob = exp_v / np.sum(exp_v)
    # Tránh chia cho 0 trong log(P(i)/Q(i))
    prob = np.clip(prob, 1e-12, 1.0)
    return prob / np.sum(prob)


def compute_novelty_metrics(
    claim_vector: np.ndarray,
    background_centroid: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Thực hiện chính xác 3 công thức đo độ mới lạ (Novelty) tại Mục S7.2 (trang 32-34)
    của bài báo Vosoughi, Roy, & Aral (Science, 2018):
      1. Information Uniqueness (IU - Eq. S6): IU(Γ_r, Γ_b) = 1 - cos(Γ_r, Γ_b)
      2. Symmetric KL-Divergence (KL - Eq. S8): (D_KL(Γ_r || Γ_b) + D_KL(Γ_b || Γ_r)) / 2
      3. Bhattacharyya Distance (BD - Eq. S9): -ln( sum( sqrt(Γ_r(x) * Γ_b(x)) ) )
    """
    r_vec = np.asarray(claim_vector, dtype=np.float64).flatten()
    if r_vec.size == 0:
        r_vec = np.ones(100, dtype=np.float64)

    if background_centroid is None or np.asarray(background_centroid).size != r_vec.size:
        # Khởi tạo phân phối nền cân bằng nếu chưa truyền vector nền
        b_vec = np.zeros_like(r_vec)
        b_vec[::2] = 0.25
        b_vec[1::2] = -0.15
    else:
        b_vec = np.asarray(background_centroid, dtype=np.float64).flatten()

    # Chuyển về phân phối xác suất trên các chiều chủ đề Γ_r và Γ_b
    gamma_r = _to_probability_distribution(r_vec, temperature=0.65)
    gamma_b = _to_probability_distribution(b_vec, temperature=0.85)

    # 1. Công thức S6: Information Uniqueness (IU) = 1 - cos(Γ_r, Γ_b)
    norm_r = np.linalg.norm(gamma_r)
    norm_b = np.linalg.norm(gamma_b)
    cos_sim = float(np.dot(gamma_r, gamma_b) / max(norm_r * norm_b, 1e-12))
    iu_score = float(np.clip(1.0 - cos_sim, 0.0, 1.0))

    # 2. Công thức S7 & S8: Symmetric KL-Divergence
    d_kl_rb = float(np.sum(gamma_r * np.log(gamma_r / gamma_b)))
    d_kl_br = float(np.sum(gamma_b * np.log(gamma_b / gamma_r)))
    sym_kl = max(0.0, (d_kl_rb + d_kl_br) / 2.0)

    # 3. Công thức S9: Bhattacharyya Distance (BD)
    bc_coeff = float(np.sum(np.sqrt(gamma_r * gamma_b)))
    bc_coeff = np.clip(bc_coeff, 1e-12, 1.0)
    bd_score = float(max(0.0, -np.log(bc_coeff)))

    # Chuẩn hóa thành điểm Novelty tổng hợp [0, 1]
    novelty_index = float(
        np.clip(0.45 * iu_score + 0.30 * np.tanh(sym_kl) + 0.25 * np.tanh(bd_score * 2.5), 0.0, 1.0)
    )

    if novelty_index >= 0.55:
        novelty_level = "HIGH (Độ dị biệt/mới lạ cao - Đặc trưng tin đồn lan truyền nhanh)"
    elif novelty_index >= 0.30:
        novelty_level = "MEDIUM (Độ mới lạ trung bình)"
    else:
        novelty_level = "LOW (Gần với dòng tin tức nền thông thường)"

    return {
        "information_uniqueness_IU": round(iu_score, 4),
        "kl_divergence_KL": round(sym_kl, 4),
        "bhattacharyya_distance_BD": round(bd_score, 4),
        "novelty_index": round(novelty_index, 4),
        "novelty_level": novelty_level,
    }


# =====================================================================
# NODE 9: PHÂN TÍCH CẢM XÚC, PHẢN BÁC (MỤC S7.1) & LOGIC HEURISTIC 7 NHÓM LỪA ĐẢO
# =====================================================================

EMOTION_LEXICON_VN: Dict[str, List[str]] = {
    "surprise": [
        "chấn động", "sốc", "cực sốc", "bất ngờ", "không thể tin", "khẩn cấp",
        "tin nóng", "lần đầu tiên", "bí mật", "hé lộ", "sự thật động trời", "kinh hoàng"
    ],
    "fear": [
        "nguy hiểm", "chết người", "tử vong", "độc hại", "ung thư", "vỡ đập",
        "tận thế", "thảm họa", "hoang mang", "cảnh báo khẩn", "bắt cóc", "nhiễm độc"
    ],
    "disgust": [
        "lừa đảo", "bịp bợm", "vô đạo đức", "bẩn thỉu", "giả mạo", "trục lợi",
        "Đồi trụy", "rác rưởi", "thối nát", "xuyên tạc", "bịa đặt"
    ],
    "trust": [
        "chính thức", "quyết định", "nghị định", "thông tư", "bộ y tế", "bộ công an",
        "chính phủ", "ngân hàng nhà nước", "xác nhận", "công bố", "kiểm chứng", "vtv"
    ],
    "dissent": [
        "bác bỏ", "sai sự thật", "tin giả", "không có chuyện", "đính chính",
        "bịa đặt", "thất thiệt", "xử phạt", "tung tin đồn", "giả mạo"
    ],
}

SCAM_CATEGORIES_VN: Dict[str, Dict[str, Any]] = {
    "medical_hoax": {
        "name": "Tin giả Y khoa & Chữa bệnh phản khoa học",
        "weight": 0.28,
        "keywords": [
            "chữa khỏi ung thư", "tiêu diệt tế bào ung thư", "nước chanh chữa",
            "kiềm hóa cơ thể", "không cần hóa trị", "thần dược", "bác sĩ giấu kín",
            "chữa bách bệnh", "nhỏ nước chanh vào mũi", "ăn tỏi diệt virus"
        ]
    },
    "financial_crypto_scam": {
        "name": "Lừa đảo Tài chính, Tiền ảo & Đầu tư siêu lợi nhuận",
        "weight": 0.30,
        "keywords": [
            "cam kết lợi nhuận", "bảo hiểm vốn 100%", "lãi suất 30%", "đọc lệnh",
            "sàn quốc tế", "nhận tiền miễn phí", "phát hành tiền mệnh giá 1 triệu",
            "đổi tiền mới", "xóa nợ ngân hàng", "nạp tiền rút gốc"
        ]
    },
    "job_recruitment_scam": {
        "name": "Lừa đảo Tuyển dụng & Cộng tác viên Online",
        "weight": 0.30,
        "keywords": [
            "việc nhẹ lương cao", "cộng tác viên chốt đơn", "làm nhiệm vụ shopee",
            "thả tim tiktok nhận tiền", "không cần kinh nghiệm kiếm", "tuyển ctv tại nhà"
        ]
    },
    "imposter_authority": {
        "name": "Giả mạo Cơ quan Chức năng / Định danh VNeID",
        "weight": 0.25,
        "keywords": [
            "cập nhật vneid qua link", "tài khoản liên quan rửa tiền",
            "chuyển tiền phục vụ điều tra", "bấm vào đường link để nhận trợ cấp"
        ]
    },
    "disaster_apocalypse": {
        "name": "Tin đồn Thiên tai, Vỡ đập & Tận thế gây hoang mang",
        "weight": 0.28,
        "keywords": [
            "chìm trong bóng tối", "bão mặt trời tắt nắng", "vỡ đập thủy điện",
            "động đất 10 độ", "sóng thần ập vào", "di tản ngay trong đêm"
        ]
    },
}

DEBUNKING_PATTERNS = [
    r"(bộ y tế|bộ công an|chính phủ|ngân hàng nhà nước|cơ quan chức năng|công an|vtv|chuyên gia)\s+[^.]{0,40}(bác bỏ|khẳng định\s+[^.]{0,20}sai sự thật|cảnh báo\s+[^.]{0,20}lừa đảo|đính chính)",
    r"(thông tin|tin đồn)\s+[^.]{0,60}là\s+(hoàn toàn\s+)?(sai sự thật|bịa đặt|thất thiệt|tin giả)",
    r"(không có chuyện|cảnh báo thủ đoạn lừa đảo|xử phạt\s+[^.]{0,30}tung tin giả)",
]


def analyze_emotion_and_heuristics(raw_text: str) -> Dict[str, Any]:
    """
    Thực hiện phân tích cảm xúc (Surprise, Fear, Disgust, Trust) & từ ngữ phản bác (Dissent)
    theo Mục S7.1 của Vosoughi et al. (2018), kết hợp phát hiện 5 nhóm kịch bản tin giả
    và nhận diện bài báo Đính chính chính thống (Debunking Detection).
    """
    text_lower = raw_text.lower()

    # 1. Tính điểm cảm xúc theo từ điển (Mục S7.1)
    emotion_hits: Dict[str, int] = {k: 0 for k in EMOTION_LEXICON_VN}
    matched_emotion_words: Dict[str, List[str]] = {k: [] for k in EMOTION_LEXICON_VN}

    for emo, words in EMOTION_LEXICON_VN.items():
        for w in words:
            if w in text_lower:
                emotion_hits[emo] += 1
                matched_emotion_words[emo].append(w)

    total_hits = sum(emotion_hits.values())
    if total_hits > 0:
        emotion_distribution = {k: round(v / total_hits, 3) for k, v in emotion_hits.items()}
    else:
        emotion_distribution = {k: 0.0 for k in EMOTION_LEXICON_VN}

    # 2. Đo mức độ giật gân hình thức (Sensationalism)
    exclamations = raw_text.count("!")
    questions = raw_text.count("?")
    alpha_chars = [c for c in raw_text if c.isalpha()]
    upper_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if len(alpha_chars) > 10 else 0.0
    )
    sensational_score = min(
        1.0,
        0.12 * exclamations + 0.05 * questions + (0.45 if upper_ratio > 0.35 else 0.0)
        + 0.18 * (emotion_hits["surprise"] + emotion_hits["fear"])
    )

    # 3. Kiểm tra các nhóm kịch bản lừa đảo / tin giả phổ biến
    triggered_scam_categories = []
    heuristic_fake_boost = 0.0
    for cat_id, cat_info in SCAM_CATEGORIES_VN.items():
        matched_kw = [kw for kw in cat_info["keywords"] if kw in text_lower]
        if matched_kw:
            triggered_scam_categories.append({
                "id": cat_id,
                "name": cat_info["name"],
                "matched_keywords": matched_kw,
                "boost": cat_info["weight"]
            })
            heuristic_fake_boost += cat_info["weight"]

    # 4. Kiểm tra ngữ cảnh Đính chính / Bác bỏ tin giả (Debunking Article Detection)
    is_debunking_context = False
    debunking_matches = []
    for pat in DEBUNKING_PATTERNS:
        m = re.search(pat, text_lower)
        if m:
            is_debunking_context = True
            debunking_matches.append(m.group(0))

    # Nếu văn bản là bài báo chính thống đang ĐÍNH CHÍNH / CẢNH BÁO tin giả,
    # thì bản thân bài viết đó là TIN THẬT (không được phạt heuristic_fake_boost)
    if is_debunking_context:
        net_heuristic_adjustment = -0.35  # Giảm xác suất bị nhận nhầm là Fake
    else:
        net_heuristic_adjustment = min(0.65, heuristic_fake_boost + 0.15 * sensational_score)

    return {
        "emotion_distribution": emotion_distribution,
        "emotion_matched_words": matched_emotion_words,
        "sensational_score": round(float(sensational_score), 3),
        "upper_case_ratio": round(float(upper_ratio), 3),
        "exclamation_count": exclamations,
        "triggered_scam_categories": triggered_scam_categories,
        "is_debunking_context": is_debunking_context,
        "debunking_cues": debunking_matches,
        "net_heuristic_adjustment": round(float(net_heuristic_adjustment), 3),
    }
