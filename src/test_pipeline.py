import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from main import predict, NewsItem


def run_verification_suite():
    samples = [
        (
            "Kịch bản 1: Tin giả Y khoa (Lan truyền tin đồn)",
            "Tin chấn động: Uống nước chanh ấm mỗi sáng hoặc ăn thực phẩm kiềm hóa cơ thể có thể tiêu diệt tế bào ung thư hoàn toàn mà không cần hóa trị!"
        ),
        (
            "Kịch bản 2: Bài báo Đính chính Chính thống (Debunking Article)",
            "Bộ Y tế chính thức bác bỏ thông tin uống nước chanh ấm chữa khỏi ung thư và khẳng định đây là tin đồn sai sự thật, phản khoa học."
        ),
        (
            "Kịch bản 3: Tin thật Chính sách Ngân hàng",
            "Ngân hàng Nhà nước quy định các giao dịch chuyển tiền trực tuyến trên 10 triệu đồng/lần hoặc tổng giá trị trên 20 triệu đồng/ngày bắt buộc phải xác thực sinh trắc học khuôn mặt."
        ),
        (
            "Kịch bản 4: Tin giả Lừa đảo Tuyển dụng Online",
            "Tuyển cộng tác viên chốt đơn Shopee, thả tim TikTok tại nhà kiếm 500.000đ đến 2.000.000đ mỗi ngày không cần kinh nghiệm."
        ),
    ]

    for title, txt in samples:
        print("=" * 85)
        print(f"TEST: {title}")
        res = predict(NewsItem(text=txt))
        hv = res["hybrid_verdict"]
        nov = res["novelty_metrics"]
        print(f"  -> Phán quyết: {hv['verdict_title']}")
        print(f"  -> Điểm xác thực (Thang 1-5 Mục S2.1): {hv['veracity_score_1_to_5']}/5.0 | Xác suất giả: {hv['final_fake_probability']*100:.1f}%")
        print(f"  -> Số mô hình ML/DL tham gia: {len(res['results'])}")
        print(f"  -> Novelty (Mục S7.2): IU={nov['information_uniqueness_IU']}, KL={nov['kl_divergence_KL']}, BD={nov['bhattacharyya_distance_BD']}")
        if hv["verified_evidences"]:
            ev = hv["verified_evidences"][0]
            print(f"  -> Bằng chứng khớp nhất: [{ev['stance']}] {ev['title']} (Sim={ev['similarity']}, Nguồn={ev['source']})")


if __name__ == "__main__":
    run_verification_suite()
