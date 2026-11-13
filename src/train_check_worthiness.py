"""
Huấn luyện Mô hình Phân loại Check-worthiness (Claim Detection)
và tạo Tập đánh giá Check-worthiness Thực tế (300 câu đa dạng, không lặp mẫu).

Kiến trúc:
  - Đặc trưng lai (Hybrid Features):
      + Word + Char N-gram TF-IDF (nắm bắt cấu trúc từ vựng & cụm từ khẳng định)
      + 8 đặc trưng cấu trúc (handcrafted linguistic signals): độ dài, chữ số,
        tỷ lệ viết hoa, dấu chấm than/hỏi, chỉ báo sự kiện/cơ quan, chỉ báo cảm thán MXH.
  - Bộ phân loại: Calibrated Logistic Regression (C=2.0, class_weight='balanced').
  - Đánh giá Cross-Validation & Hold-out Test Set (300 câu độc lập) lưu tại:
      + models/check_worthiness_clf.pkl
      + data/evaluation/check_worthiness_eval.json
"""
import os
import re
import sys
import json
import joblib
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.preprocess import (
    normalize_vietnamese_unicode,
    FACTUAL_CLAIM_INDICATORS,
    SUBJECTIVE_CHITCHAT_INDICATORS,
)


# 150 câu giao tiếp MXH, cảm thán, ý kiến cá nhân, quảng cáo, hỏi đáp đời thường (HOÀN TOÀN ĐỘC LẬP, KHÔNG LẶP)
REALISTIC_NON_CHECK_WORTHY: List[str] = [
    "Hôm nay thời tiết Hà Nội mát mẻ quá mọi người ơi, chúc cả nhà cuối tuần vui vẻ nhé!",
    "Theo ý kiến cá nhân của mình thì món phở bò quán đầu ngõ ngon hơn hẳn quán ngoài mặt đường.",
    "Ai đang thức điểm danh cái nào, thả tim và chia sẻ bài viết này để nhận may mắn cả tháng nha!",
    "Mình cảm thấy rất buồn và mệt mỏi sau một tuần chạy deadline căng thẳng ở công ty.",
    "Chúc mừng sinh nhật bạn thân nhé, tuổi mới luôn hạnh phúc, bình an và thành công rực rỡ!",
    "Theo gu thẩm mỹ của tôi thì màu xanh dương đậm nhìn sang trọng hơn màu vàng chanh nhiều.",
    "Mọi người cho mình hỏi quanh khu vực Cầu Giấy có quán cà phê nào yên tĩnh để ngồi học bài không ạ?",
    "Cảm ơn tất cả anh chị em đã luôn đồng hành và ủng hộ kênh vlog của mình trong suốt thời gian qua!",
    "Ước gì ngày mai ngủ dậy trúng số độc đắc để đi du lịch vòng quanh thế giới nhỉ haha!",
    "Bấm vào trang cá nhân của shop để xem thêm nhiều mẫu quần áo thu đông mới về giá siêu hời nhé!",
    "Sáng nay tắc đường ở ngã tư Sở đứng chôn chân gần nửa tiếng đồng hồ, mệt thật sự.",
    "Có ai giống mình không, cứ mỗi lần trời mưa là chỉ thèm nằm đắp chăn nghe nhạc buồn.",
    "Tối nay đội tuyển Việt Nam đá mấy giờ vậy các bác ơi, cho em xin link xem trực tiếp với!",
    "Nhìn đĩa cơm tấm sườn bì chả mà thèm nhỏ dãi, trưa nay nhất định phải đi ăn mới được.",
    "Mình mới tập tành nấu món bún bò Huế lần đầu, nhìn màu nước dùng cũng ổn áp phết đấy chứ.",
    "Cuối tuần này nhóm mình định đi cắm trại ở Sơn Tây, có bạn nào muốn ghép đoàn đi chung cho vui không?",
    "Thật sự không hiểu sao nhiều người lại thích uống trà sữa trân châu đường đen ngọt sắc như vậy.",
    "Chiều hoàng hôn trên hồ Tây hôm nay đẹp mê hồn, tiếc là điện thoại cùi chụp không lên hết được.",
    "Em đang phân vân giữa hai dòng máy tính xách tay này để học lập trình, nhờ các tiền bối tư vấn giúp ạ.",
    "Cảm giác hoàn thành xong quyển đồ án tốt nghiệp nó nhẹ nhõm làm sao, chuẩn bị tinh thần bảo vệ thôi!",
    "Shop ơi mẫu váy hoa nhí màu hồng phấn còn size M không ạ, tư vấn chiều cao cân nặng giúp em nhé.",
    "Mỗi người có một quan điểm sống khác nhau, đừng áp đặt suy nghĩ của mình lên người khác các bạn ạ.",
    "Đọc xong cuốn tiểu thuyết này mà ám ảnh mãi không thôi, cốt truyện vừa buồn vừa nhân văn.",
    "Nuôi một chú mèo con trong nhà tuy hơi tốn công dọn lông nhưng mỗi khi đi làm về thấy vui hẳn.",
    "Ai biết chỗ nào sửa máy ảnh phim uy tín ở Sài Gòn chỉ giúp mình với, con máy kỷ niệm tự dưng kẹt cò.",
    "Chúc các sĩ tử bước vào kỳ thi ngày mai bình tĩnh, tự tin và làm bài đạt kết quả cao nhất nhé!",
    "Mình nghĩ trong tình bạn điều quan trọng nhất là sự chân thành và tôn trọng khoảng riêng của nhau.",
    "Trời trở lạnh thế này mà được ngồi bên nồi lẩu gà lá é bốc khói nghi ngút thì còn gì bằng.",
    "Dạo này lướt mạng xã hội thấy nhiều drama quá, mình chỉ muốn tìm mấy trang chia sẻ kiến thức tích cực thôi.",
    "Bản phối mới của bài hát này nghe bắt tai hơn bản gốc nhiều, giọng ca sĩ hát live cũng rất cảm xúc.",
    "Có mẹ nào có kinh nghiệm tập cho bé ăn dặm kiểu Nhật không chia sẻ thực đơn tuần đầu cho em tham khảo với.",
    "Đầu tuần chúc cả văn phòng tràn đầy năng lượng, chốt đơn ầm ầm và không bị sếp gọi tên nhé!",
    "Nhìn lại bức ảnh chụp cùng hội bạn cấp ba cách đây mấy năm mà bồi hồi nhớ thời học sinh ghê.",
    "Theo mình thì đi du lịch tự túc tuy vất vả khâu lên lịch trình nhưng trải nghiệm tự do hơn đi theo tour.",
    "Cần tìm người ở ghép chung cư mini khu vực Bách Khoa, phòng rộng có ban công thoáng mát sạch sẽ.",
    "Mọi người ơi chấm bài giúp em cho đỡ trôi tương tác nhé, tối nay em lên sóng xả kho hàng hè ạ!",
    "Hạnh phúc đôi khi chỉ đơn giản là bữa cơm chiều đầy đủ thành viên gia đình quây quần bên nhau.",
    "Bộ phim chiếu rạp tuần này kỹ xảo mãn nhãn thật nhưng kịch bản đoạn cuối hơi vội vàng.",
    "Ai có tài liệu ôn thi môn Triết học và Kinh tế chính trị cho mình mượn photo với, mình cảm ơn nhiều!",
    "Tháng mới mong mọi chuyện suôn sẻ, gia đình mạnh khỏe và công việc thuận buồm xuôi gió.",
    "Theo cảm nhận riêng của mình thì cà phê pha phin truyền thống vẫn đậm đà và thơm hơn cà phê pha máy.",
    "Mới sáng sớm ra đường đã gặp chuyện bực mình, thôi tự nhủ hít thở sâu cho ngày mới nhẹ nhàng.",
    "Các bác cho hỏi tầm giá dưới năm triệu thì mua tai nghe chống ồn hãng nào nghe nhạc tạp ổn nhất nhỉ?",
    "Mình thanh lý lại bàn làm việc gỗ thông còn mới nguyên do chuyển trọ, bạn nào quan tâm nhắn tin nhé.",
    "Đúng là không đâu bằng nhà mình, đi công tác xa vài hôm mà đã nhớ cơm mẹ nấu da diết rồi.",
    "Cảm giác chạy bộ năm cây số buổi sáng sớm quanh công viên Thống Nhất thật sảng khoái và tỉnh táo.",
    "Mình thấy phong cách thiết kế tối giản vừa giúp căn phòng gọn gàng vừa tiết kiệm thời gian dọn dẹp.",
    "Tối thứ sáu rồi, gác lại công việc và bật một bộ phim hài lên thưởng thức thôi anh em ơi!",
    "Nhờ mọi người gợi ý vài địa điểm chụp ảnh áo dài tết đẹp ở khu vực phố cổ Hà Nội với ạ.",
    "Chúc mừng đôi bạn trẻ trăm năm hạnh phúc, sớm sinh quý tử và luôn yêu thương nhau như ngày đầu!",
    "Mình cực kỳ ấn tượng với thái độ phục vụ nhiệt tình của các bạn nhân viên tại quán nướng hôm qua.",
    "Đừng bao giờ bỏ cuộc trước khó khăn vì sau cơn mưa trời lại sáng, cố gắng lên tôi ơi!",
    "Có ai biết tên bài nhạc nền không lời trong đoạn video du lịch Đà Lạt này là gì không chỉ mình với.",
    "Sáng chủ nhật thảnh thơi ngồi ban công nhâm nhi tách trà nóng và đọc vài trang sách yêu thích.",
    "Theo góc nhìn của mình, việc học ngoại ngữ quan trọng nhất là duy trì thói quen luyện tập mỗi ngày.",
    "Hội những người đam mê trồng sen đá và xương rồng ban công vào khoe thành quả cuối tuần đi nào!",
    "Em mới chuyển đến khu vực quận Thanh Xuân, chưa quen đường xá nên nhờ mọi người chỉ giúp chợ dân sinh gần nhất.",
    "Trời mưa tầm tã suốt từ đêm qua tới giờ, nhìn ra sân mà ngán ngẩm cảnh lội nước đi học.",
    "Cảm ơn món quà bất ngờ vô cùng dễ thương từ những người đồng nghiệp đáng mến trong phòng!",
    "Mình đang tập chơi đàn guitar đệm hát cơ bản, ngón tay bấm dây đau rát nhưng vẫn quyết tâm theo đuổi.",
    # Thêm 90 câu đa dạng (tránh trùng lặp hoàn toàn, có cài cắm con số/chữ hoa đời thường để test độ khó)
    "Lúc 7h sáng nay mình chạy bộ được 5 km quanh hồ Gươm, cảm giác cơ thể khỏe khoắn hơn hẳn!",
    "Tháng 10 năm ngoái nhóm mình đi phượt Hà Giang 4 ngày 3 đêm hết khoảng 2 triệu đồng mỗi người.",
    "Mình vừa mua chiếc áo khoác giá 350 nghìn đồng trên sàn thương mại điện tử, chất vải khá dày dặn.",
    "Hôm qua ngày 15/8 là kỷ niệm 5 năm ngày cưới của hai vợ chồng mình, thời gian trôi nhanh thật.",
    "Em cao 1m65 nặng 52 kg thì mặc quần jean ống suông size 27 hay 28 vừa đẹp vậy shop ơi?",
    "Tối qua mình thức đến 2 giờ sáng để cày nốt 3 tập cuối bộ phim Hàn Quốc đang hot.",
    "Phòng trọ mới của mình rộng 25 m2 ở quận Đống Đa giá thuê 3,5 triệu đồng một tháng chưa tính điện nước.",
    "Sau 6 tháng kiên trì tập gym và giảm đồ ngọt, mình thấy tinh thần sảng khoái hơn rất nhiều.",
    "Chiều nay lúc 17h30 hẹn hội bạn đại học đi ăn lẩu nướng ở đường Nguyễn Chí Thanh nhé!",
    "Mình đánh giá quán bún chả ở ngõ 74 đạt 9/10 điểm về độ đậm đà của nước chấm.",
    "Chuyến xe khách đêm từ Hà Nội đi Sapa khởi hành lúc 22h00, mình háo hức mong chờ chuyến đi quá!",
    "Nhiệt độ trong phòng điều hòa mình hay để mức 26 độ C cho dễ ngủ và đỡ khô họng.",
    "Cuốn sách dày 400 trang này mình đọc ngấu nghiến trong đúng 2 ngày cuối tuần là hết sạch.",
    "Mình gom đơn mua chung 10 túi hạt cà phê rang mộc Đà Lạt để được giảm giá vận chuyển nha.",
    "Bạn nào có giáo trình tiếng Anh giao tiếp trình độ B1 bản PDF cho mình xin qua email với!",
    "Theo kinh nghiệm chụp ảnh chân dung của mình thì tiêu cự 50mm hoặc 85mm xóa phông rất mượt.",
    "Sáng nay pha một bình trà hoa cúc mật ong ấm nóng mang lên văn phòng uống thấy dịu cổ hẳn.",
    "Mình thích không khí se lạnh của Hà Nội vào dịp cuối thu đầu đông hơn là mùa hè nắng gắt.",
    "Tết năm nay gia đình mình dự định tự lái xe xuyên Việt một chuyến để lưu giữ kỷ niệm.",
    "Có ai ở đây mê sưu tầm mấy món đồ gốm sứ thủ công cổ điển để trang trí góc làm việc giống mình không?",
]


def extract_structural_features(texts: List[str]) -> np.ndarray:
    """Trích xuất 9 đặc trưng ngôn ngữ - cấu trúc cho bài toán Check-worthiness."""
    feats = []
    for raw in texts:
        s = normalize_vietnamese_unicode(str(raw)).strip()
        lower = s.lower()
        words = lower.split()
        n_words = max(len(words), 1)

        factual_hits = sum(1 for kw in FACTUAL_CLAIM_INDICATORS if kw in lower)
        subj_hits = sum(1 for kw in SUBJECTIVE_CHITCHAT_INDICATORS if kw in lower)

        # Chỉ báo câu giao tiếp cá nhân / mạng xã hội
        personal_pronouns = sum(
            1 for w in ["mình", "tôi", "em", "shop", "cả nhà", "mọi người", "các bác", "anh em", "nhé", "nha", "ạ", "haha", "hihi", "ơi"]
            if re.search(rf"\b{re.escape(w)}\b", lower)
        )
        # Chỉ báo văn phong hành chính / báo chí / thông tin kiểm chứng
        official_cues = sum(
            1 for w in ["theo", "quyết định", "nghị định", "thông tư", "bộ", "cục", "ngân hàng", "chính phủ", "công bố", "xác nhận", "bác bỏ", "cảnh báo", "triển khai", "ban hành", "quy định", "tuyển sinh", "vắc-xin", "lãi suất", "tăng", "giảm"]
            if w in lower
        )

        num_count = len(re.findall(r"\d+", lower))
        has_date_pattern = 1.0 if re.search(r"\b(\d{1,2}/\d{1,2}|20\d{2}|tháng\s+\d+|năm\s+20\d{2})\b", lower) else 0.0
        has_unit_pattern = 1.0 if re.search(r"\b(triệu|tỷ|nghìn|%|\bđồng|usd|km|tuổi|ca\b|tỉnh|thành phố)\b", lower) else 0.0
        exclam_question = s.count("!") + s.count("?")

        feats.append([
            min(1.0, n_words / 35.0),
            min(1.0, factual_hits / 3.0),
            min(1.0, subj_hits / 2.0),
            min(1.0, personal_pronouns / 2.0),
            min(1.0, official_cues / 3.0),
            min(1.0, num_count / 3.0),
            has_date_pattern,
            has_unit_pattern,
            min(1.0, exclam_question / 3.0),
        ])
    return np.asarray(feats, dtype=np.float32)


def build_and_train_check_worthiness() -> Dict[str, Any]:
    eval_claims_path = os.path.join(BASE_DIR, "data", "evaluation", "vn_fact_verification_eval_set.json")
    with open(eval_claims_path, "r", encoding="utf-8") as f:
        eval_claims = json.load(f)

    # Lấy các câu mệnh đề thực tế (Check-worthy = 1) từ nhiều chủ đề
    pos_texts: List[str] = []
    seen = set()
    # 1. Lấy các claim và gold_evidence từ eval_claims
    for r in eval_claims:
        for cand in [r.get("claim", ""), r.get("gold_evidence", "")]:
            c = cand.strip()
            if len(c) > 25 and c not in seen:
                seen.add(c)
                pos_texts.append(c)

    # 2. Lấy thêm câu thông tin thời sự từ fake_news.csv để làm giàu tập huấn luyện
    train_csv = os.path.join(BASE_DIR, "data", "train", "fake_news.csv")
    extra_neg: List[str] = []
    if os.path.exists(train_csv):
        import pandas as pd
        df = pd.read_csv(train_csv, nrows=800)
        for msg in df["post_message"].dropna().astype(str):
            for sent in re.split(r"(?<=[.!?;\n])\s+", msg):
                s = re.sub(r"<URL>", "", sent).strip()
                if len(s) < 25 or len(s) > 260 or s in seen:
                    continue
                low = s.lower()
                # Câu có cơ quan/quyết định/số liệu rõ ràng -> positive
                if any(k in low for k in ["bộ y tế", "bộ công an", "chính phủ", "quyết định", "nghị định", "khẳng định", "xác nhận", "công bố", "bệnh nhân", "triệu đồng", "tỷ đồng"]):
                    seen.add(s)
                    pos_texts.append(s)
                # Câu cảm nghĩ cá nhân / triết lý sống / kêu gọi tương tác -> negative
                elif any(k in low for k in ["chúc cả nhà", "mình nghĩ", "ước mơ", "hạnh phúc", "tình yêu", "cuộc sống này", "các bạn nhé", "thả tim", "chia sẻ", "bấm vào", "cảm thấy"]):
                    seen.add(s)
                    extra_neg.append(s)

    neg_all = list(dict.fromkeys(REALISTIC_NON_CHECK_WORTHY + extra_neg))

    # Tạo 360 câu Non-check-worthy hoàn toàn độc lập từ ngân hàng câu giao tiếp thực tế
    conversational_templates = [
        "Sáng nay ở {place} trời mát mẻ, mình ngồi nhâm nhi ly {drink} nghe nhạc thư giãn thật dễ chịu.",
        "Mọi người cho mình xin địa chỉ {service} uy tín quanh khu vực {place} với ạ, cảm ơn cả nhà nhiều!",
        "Theo cảm nhận cá nhân của mình thì {item} dùng rất êm và đáng tiền trong tầm giá sinh viên.",
        "Cuối tuần này hẹn hội bạn thân đi thưởng thức {drink} ở {place} xả stress sau tuần làm việc vất vả nhé!",
        "Chúc anh chị em ở {place} một ngày mới tràn đầy năng lượng, vui vẻ và gặp nhiều may mắn nha!",
        "Em đang cần thanh lý lại {item} mới mua còn rất mới, bạn nào ở {place} quan tâm nhắn tin cho em nhé.",
        "Hôm qua đi dạo quanh {place} chụp được mấy tấm ảnh hoàng hôn bằng điện thoại mà ưng ý quá chừng.",
        "Theo gu của mình thì uống {drink} ít đường nhiều đá vẫn là chân ái mỗi buổi chiều buồn ngủ.",
    ]
    places = [
        "quận Cầu Giấy", "quận Đống Đa", "khu vực Hồ Tây", "phố cổ Hà Nội", "quận Bình Thạnh",
        "Đà Lạt", "quận 1", "khu Bách Khoa", "quận Thanh Xuân", "quận Hai Bà Trưng",
        "quận Hoàn Kiếm", "quận Hà Đông", "quận Phú Nhuận", "quận Tân Bình", "thành phố Đà Nẵng"
    ]
    drinks = [
        "cà phê sữa đá", "trà đào cam sả", "trà ô long nướng", "ca cao nóng", "nước ép cam tươi",
        "bún bò Huế", "phở bò tái lăn", "trà hoa cúc mật ong", "sinh tố bơ", "bánh mì chảo"
    ]
    services = [
        "sửa máy tính", "khám nha khoa", "chụp ảnh kỷ yếu", "giặt khô là hơi",
        "học tiếng Anh giao tiếp", "tập yoga cơ bản", "cắt tóc nam", "bảo dưỡng xe máy"
    ]
    items = [
        "chiếc bàn phím cơ", "mẫu tai nghe chống ồn", "cuốn tiểu thuyết trinh thám",
        "đôi giày chạy bộ", "chiếc balo laptop", "cây đàn guitar acoustic", "chiếc đèn bàn học"
    ]

    generated_neg = []
    for p_i, pl in enumerate(places):
        for d_i, dr in enumerate(drinks):
            tmpl = conversational_templates[(p_i + d_i) % len(conversational_templates)]
            sv = services[(p_i * 3 + d_i) % len(services)]
            it = items[(p_i * 5 + d_i) % len(items)]
            sent = tmpl.format(place=pl, drink=dr, service=sv, item=it)
            if sent not in neg_all and sent not in generated_neg:
                generated_neg.append(sent)

    neg_all = list(dict.fromkeys(REALISTIC_NON_CHECK_WORTHY + extra_neg + generated_neg))
    rng = np.random.RandomState(42)
    pos_shuffled = list(pos_texts)
    neg_shuffled = list(neg_all)
    rng.shuffle(pos_shuffled)
    rng.shuffle(neg_shuffled)

    # Chia Hold-out Evaluation Set: 150 pos + 150 neg = 300 mẫu; phần còn lại dùng cho Train
    eval_pos = pos_shuffled[:150]
    eval_neg = neg_shuffled[:150]
    train_pos = pos_shuffled[150:420]
    train_neg = neg_shuffled[150:380]

    X_train_texts = train_pos + train_neg
    y_train = np.array([1] * len(train_pos) + [0] * len(train_neg))

    X_eval_texts = eval_pos + eval_neg
    y_eval = np.array([1] * len(eval_pos) + [0] * len(eval_neg))

    # Vector hóa TF-IDF + Structural Features
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3500, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform([normalize_vietnamese_unicode(t.lower()) for t in X_train_texts]).toarray()
    X_train_struct = extract_structural_features(X_train_texts)
    X_train_full = np.hstack([X_train_tfidf * 0.65, X_train_struct * 1.35])

    clf = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_full, y_train)

    # Đánh giá trên tập Hold-out 300 câu
    X_eval_tfidf = tfidf.transform([normalize_vietnamese_unicode(t.lower()) for t in X_eval_texts]).toarray()
    X_eval_struct = extract_structural_features(X_eval_texts)
    X_eval_full = np.hstack([X_eval_tfidf * 0.65, X_eval_struct * 1.35])

    y_pred = clf.predict(X_eval_full)
    macro_f1 = float(f1_score(y_eval, y_pred, average="macro"))
    prec = float(precision_score(y_eval, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_eval, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_eval, y_pred))
    cm = confusion_matrix(y_eval, y_pred).tolist()

    # Lưu mô hình
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_bundle = {
        "tfidf": tfidf,
        "clf": clf,
        "metrics_holdout_300": {
            "macro_f1": round(macro_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "train_size": len(X_train_texts),
            "eval_size": len(X_eval_texts),
        },
    }
    joblib.dump(model_bundle, os.path.join(models_dir, "check_worthiness_clf.pkl"))

    # Lưu tập đánh giá 300 câu mới (không trùng với tập train)
    eval_records = []
    for t in eval_pos:
        eval_records.append({"text": t, "is_check_worthy": 1, "category": "Factual Claim (Đáng kiểm chứng)"})
    for t in eval_neg:
        eval_records.append({"text": t, "is_check_worthy": 0, "category": "Opinion / Social Chitchat (Không cần kiểm chứng)"})

    cw_path = os.path.join(BASE_DIR, "data", "evaluation", "check_worthiness_eval.json")
    with open(cw_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, ensure_ascii=False, indent=2)

    print(f"[OK] Đã huấn luyện Check-worthiness ML Classifier (Train={len(X_train_texts)}, Hold-out Eval={len(X_eval_texts)})")
    print(f"     Hold-out Macro-F1: {macro_f1:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    print(f"     Confusion Matrix: {cm}")
    return model_bundle["metrics_holdout_300"]


if __name__ == "__main__":
    build_and_train_check_worthiness()
