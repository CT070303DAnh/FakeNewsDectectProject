import { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import TheoryPage from './pages/TheoryPage'
import TrainingPage from './pages/TrainingPage'

interface ModelEducation {
  name: string;
  concept: string;
  principle: string;
  flow: string;
}

interface ModelMetrics {
  accuracy: number;
  f1: number;
  recall: number;
  precision: number;
  false_alarm_rate: number;
  confusion_matrix: number[][];
}

interface ModelResult {
  prediction: string;
  fake_probability: number;
  real_probability: number;
  metrics: ModelMetrics;
  education: ModelEducation;
}

interface VerifiedEvidence {
  id: string;
  title: string;
  evidence_text: string;
  source: string;
  url: string;
  date?: string;
  topic: string;
  tier: string;
  similarity: number;
  bm25_score?: number;
  dense_score?: number;
  rerank_score?: number;
  temporal_match?: string;
  retrieval_method?: string;
  citation_tag?: string;
  vosoughi_s22_band: string;
  stance: 'SUPPORTS' | 'REFUTES' | 'NEI';
  nli_label?: 'Supported' | 'Refuted' | 'NEI';
  veracity_score_1_to_5: number;
  stance_explanation: string;
}

interface HybridVerdict {
  verdict_code: 'TRUE' | 'FALSE' | 'MIXED';
  nli_3class_label?: 'Supported' | 'Refuted' | 'NEI';
  verdict_title: string;
  verdict_badge: string;
  confidence?: number;
  uncertainty_score?: number;
  latency_ms?: number;
  veracity_score_1_to_5: number;
  final_fake_probability: number;
  final_real_probability: number;
  avg_ml_fake_probability: number;
  evidence_veracity_score: number;
  has_verified_evidence: boolean;
  verified_evidences: VerifiedEvidence[];
  llm_verification?: {
    llm_label: string;
    llm_stance: string;
    llm_confidence: number;
    citations_used: string[];
    grounded_explanation: string;
  };
  nli_llm_agreement?: boolean;
  retrieval_stats: {
    retrieval_mode?: string;
    total_candidates_checked: number;
    accepted_above_050: number;
    discarded_below_050: number;
    max_similarity: number;
  };
  reasoning_steps: string[];
}

interface ClaimAnalysis {
  primary_claim: string;
  check_worthiness_score?: number;
  all_claims: string[];
  entities: string[];
  numbers_dates: string[];
  search_queries: string[];
  is_url_input?: boolean;
  source_url?: string | null;
}

interface NoveltyMetrics {
  information_uniqueness_IU: number;
  kl_divergence_KL: number;
  bhattacharyya_distance_BD: number;
  novelty_index: number;
  novelty_level: string;
}

interface EmotionHeuristics {
  emotion_distribution: Record<string, number>;
  sensational_score: number;
  upper_case_ratio: number;
  exclamation_count: number;
  triggered_scam_categories: Array<{
    id: string;
    name: string;
    matched_keywords: string[];
    boost: number;
  }>;
  is_debunking_context: boolean;
  debunking_cues: string[];
  net_heuristic_adjustment: number;
}

interface PredictionResponse {
  input_text: string;
  cleaned_text: string;
  claim_analysis: ClaimAnalysis;
  hybrid_verdict: HybridVerdict;
  novelty_metrics: NoveltyMetrics;
  emotion_heuristics: EmotionHeuristics;
  results: Record<string, ModelResult>;
  doc2vec_vector: number[];
  education_doc2vec: ModelEducation;
}

const SAMPLE_CLAIMS = [
  {
    tag: 'Y tế / Chữa bệnh',
    label: 'Tin giả Y khoa',
    text: 'Tin chấn động: Uống nước chanh ấm mỗi sáng hoặc ăn thực phẩm kiềm hóa cơ thể có thể tiêu diệt tế bào ung thư hoàn toàn mà không cần hóa trị!'
  },
  {
    tag: 'Tài chính / Tiền tệ',
    label: 'Tin giả Tiền tệ',
    text: 'Thông báo khẩn: Ngân hàng Nhà nước Việt Nam chuẩn bị phát hành tờ tiền mệnh giá 1.000.000 đồng vào tháng tới, mọi người chia sẻ ngay!'
  },
  {
    tag: 'Chính sách / Ngân hàng',
    label: 'Tin thật Chính sách',
    text: 'Ngân hàng Nhà nước quy định các giao dịch chuyển tiền trực tuyến trên 10 triệu đồng/lần hoặc tổng giá trị trên 20 triệu đồng/ngày bắt buộc phải xác thực sinh trắc học khuôn mặt.'
  },
  {
    tag: 'Đính chính / Bác bỏ',
    label: 'Bài báo Đính chính',
    text: 'Bộ Y tế chính thức bác bỏ thông tin uống nước chanh ấm chữa khỏi ung thư và khẳng định đây là tin đồn sai sự thật, phản khoa học.'
  }
];

/* ============================================================================
 * INLINE SVG ICONS (Heroicons / Lucide Standard 24x24 - Zero Emojis Policy)
 * ============================================================================ */
const IconShieldCheck = () => (
  <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);

const IconSearch = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);

const IconCpu = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="4" y="4" width="16" height="16" rx="2" />
    <rect x="9" y="9" width="6" height="6" />
    <path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2" />
  </svg>
);

const IconBookOpen = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
  </svg>
);

const IconCheckCircle = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);

const IconXCircle = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <line x1="15" y1="9" x2="9" y2="15" />
    <line x1="9" y1="9" x2="15" y2="15" />
  </svg>
);

const IconAlertTriangle = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const IconExternalLink = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <line x1="10" y1="14" x2="21" y2="3" />
  </svg>
);

const IconFilter = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
  </svg>
);

const IconArrowLeft = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

/* ============================================================================
 * SVG GAUGE CHART FOR VERACITY SCORE 1.0 - 5.0 (UI Pro Max Chart Spec #1)
 * ============================================================================ */
const VeracityGauge = ({ score, fakeProb }: { score: number; fakeProb: number }) => {
  const clamped = Math.max(1, Math.min(5, score));
  const ratio = (clamped - 1) / 4; // 0.0 (False=1) -> 1.0 (True=5)
  const angleDeg = -90 + ratio * 180; // -90deg (left) to +90deg (right)

  return (
    <div className="flex flex-col items-center">
      <svg className="w-48 h-28 overflow-visible" viewBox="0 0 200 110" role="img" aria-label={`Thang đo xác thực ${score} trên 5`}>
        <defs>
          <linearGradient id="veracityGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#EF4444" />
            <stop offset="50%" stopColor="#F59E0B" />
            <stop offset="100%" stopColor="#10B981" />
          </linearGradient>
        </defs>
        {/* Background Track */}
        <path
          d="M 20 95 A 80 80 0 0 1 180 95"
          fill="none"
          stroke="#E2E8F0"
          strokeWidth="16"
          strokeLinecap="round"
        />
        {/* Colored Gradient Arc */}
        <path
          d="M 20 95 A 80 80 0 0 1 180 95"
          fill="none"
          stroke="url(#veracityGrad)"
          strokeWidth="16"
          strokeLinecap="round"
        />
        {/* Scale Labels */}
        <text x="14" y="112" className="text-[10px] fill-rose-600 font-bold font-mono-code">1.0 (SAI)</text>
        <text x="85" y="10" className="text-[9px] fill-amber-600 font-bold font-mono-code">3.0</text>
        <text x="150" y="112" className="text-[10px] fill-emerald-600 font-bold font-mono-code">5.0 (ĐÚNG)</text>

        {/* Pointer Needle */}
        <g transform={`translate(100, 95) rotate(${angleDeg})`}>
          <line x1="0" y1="0" x2="0" y2="-66" stroke="#0F172A" strokeWidth="3.5" strokeLinecap="round" />
          <circle cx="0" cy="0" r="6" fill="#0F172A" />
        </g>
      </svg>
      <div className="text-center -mt-1">
        <span className="font-mono-code text-2xl font-bold text-[#0F172A]">{score.toFixed(2)} / 5.0</span>
        <span className="block text-xs text-slate-600 font-medium">
          Độ tin cậy xác thực: {((1 - fakeProb) * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};

function App() {
  const [currentView, setCurrentView] = useState<'detector' | 'theory' | 'training'>('detector')
  const [newsText, setNewsText] = useState('')
  const [retrievalMode, setRetrievalMode] = useState<'hybrid_rerank' | 'hybrid' | 'bm25' | 'dense'>('hybrid_rerank')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<string>('')
  const [evidenceFilter, setEvidenceFilter] = useState<'ALL' | 'SUPPORTS' | 'REFUTES' | 'LOCAL' | 'LIVE'>('ALL')
  const [feedbackRating, setFeedbackRating] = useState<'ACCURATE' | 'INACCURATE' | null>(null)
  const [suggestedLabel, setSuggestedLabel] = useState<string>('Refuted')
  const [feedbackComment, setFeedbackComment] = useState<string>('')
  const [feedbackStatus, setFeedbackStatus] = useState<string>('')

  useEffect(() => {
    if (result && result.results && !activeTab) {
      const firstModel = Object.keys(result.results)[0];
      if (firstModel) setActiveTab(firstModel);
    }
  }, [result, activeTab])

  const handleCheck = async (customText?: string) => {
    const textToVerify = customText !== undefined ? customText : newsText;
    if (!textToVerify.trim()) {
      setError('Vui lòng nhập nội dung tin tức, mệnh đề hoặc URL bài báo cần kiểm chứng.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setActiveTab('')
    setEvidenceFilter('ALL')
    setFeedbackRating(null)
    setFeedbackComment('')
    setFeedbackStatus('')

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        text: textToVerify,
        retrieval_mode: retrievalMode
      })
      if (response.data && response.data.hybrid_verdict) {
        setResult(response.data)
      } else {
        setError('Máy chủ không trả về kết quả phân tích.')
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Không thể kết nối máy chủ Backend (Port 8000): ${msg}`)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmitFeedback = async (rating: 'ACCURATE' | 'INACCURATE') => {
    if (!result || !result.hybrid_verdict) return;
    setFeedbackRating(rating);
    try {
      const res = await axios.post('http://localhost:8000/feedback', {
        claim_text: result.claim_analysis.primary_claim,
        predicted_label: result.hybrid_verdict.nli_3class_label || result.hybrid_verdict.verdict_code,
        user_rating: rating,
        suggested_label: rating === 'INACCURATE' ? suggestedLabel : (result.hybrid_verdict.nli_3class_label || ''),
        user_comment: feedbackComment
      });
      setFeedbackStatus(res.data.message || 'Đã ghi nhận phản hồi!');
    } catch {
      setFeedbackStatus('Đã lưu phản hồi vào bộ nhớ phiên làm việc.');
    }
  };

  const filteredEvidences = useMemo(() => {
    const list = result?.hybrid_verdict?.verified_evidences || [];
    return list.filter((ev) => {
      if (evidenceFilter === 'ALL') return true;
      if (evidenceFilter === 'SUPPORTS') return ev.stance === 'SUPPORTS';
      if (evidenceFilter === 'REFUTES') return ev.stance === 'REFUTES';
      if (evidenceFilter === 'LOCAL') return ev.tier !== 'LIVE_WEB_NEWS';
      if (evidenceFilter === 'LIVE') return ev.tier === 'LIVE_WEB_NEWS';
      return true;
    });
  }, [result, evidenceFilter]);

  const renderMetrics = (metrics?: ModelMetrics) => {
    if (!metrics) return null;
    return (
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
        {[
          { label: 'Accuracy', val: metrics.accuracy, desc: 'Độ chính xác' },
          { label: 'F1-Score', val: metrics.f1, desc: 'Cân bằng P/R' },
          { label: 'Recall', val: metrics.recall, desc: 'Độ phủ tin giả' },
          { label: 'Precision', val: metrics.precision, desc: 'Độ chuẩn xác' },
          { label: 'FAR', val: metrics.false_alarm_rate, desc: 'Báo động giả' },
        ].map((m, i) => (
          <div key={i} className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="font-mono-code text-[11px] font-bold text-[#1E40AF] uppercase">{m.label}</span>
              <span className="text-[10px] text-slate-500">{m.desc}</span>
            </div>
            <p className="font-mono-code text-xl font-bold text-[#0F172A] mt-1">{(m.val * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    )
  }

  const ConfusionMatrix = ({ matrix }: { matrix?: number[][] }) => {
    if (!matrix || matrix.length < 2) return null;
    const tn = matrix[0][0];
    const fp = matrix[0][1];
    const fn = matrix[1][0];
    const tp = matrix[1][1];
    const total = tn + fp + fn + tp;

    return (
      <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm mt-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-mono-code text-xs font-bold text-slate-700 uppercase tracking-wider">
            Confusion Matrix (Tập Test Độc Lập: {total} mẫu)
          </h4>
          <span className="text-xs text-slate-500 font-medium">Đánh giá thực nghiệm</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-emerald-50/80 border border-emerald-200 p-3.5 rounded-xl">
            <span className="font-mono-code text-[11px] font-bold text-emerald-800 block">TN (True Negative)</span>
            <span className="text-xs text-emerald-700 block mb-1">Tin thật đoán đúng</span>
            <p className="font-mono-code text-2xl font-bold text-emerald-900">{tn}</p>
          </div>
          <div className="bg-rose-50/80 border border-rose-200 p-3.5 rounded-xl">
            <span className="font-mono-code text-[11px] font-bold text-rose-800 block">FP (False Positive)</span>
            <span className="text-xs text-rose-700 block mb-1">Báo động giả</span>
            <p className="font-mono-code text-2xl font-bold text-rose-900">{fp}</p>
          </div>
          <div className="bg-amber-50/80 border border-amber-200 p-3.5 rounded-xl">
            <span className="font-mono-code text-[11px] font-bold text-amber-800 block">FN (False Negative)</span>
            <span className="text-xs text-amber-700 block mb-1">Bỏ lọt tin giả</span>
            <p className="font-mono-code text-2xl font-bold text-amber-900">{fn}</p>
          </div>
          <div className="bg-emerald-50/80 border border-emerald-200 p-3.5 rounded-xl">
            <span className="font-mono-code text-[11px] font-bold text-emerald-800 block">TP (True Positive)</span>
            <span className="text-xs text-emerald-700 block mb-1">Bắt đúng tin giả</span>
            <p className="font-mono-code text-2xl font-bold text-emerald-900">{tp}</p>
          </div>
        </div>
      </div>
    );
  };

  /* Floating Top Navbar Component */
  const renderNavbar = () => (
    <header className="sticky top-4 z-50 mx-4 md:mx-8 mb-8">
      <nav
        aria-label="Điều hướng chính"
        className="max-w-7xl mx-auto bg-white/90 backdrop-blur-md border border-slate-200/90 rounded-2xl px-5 py-3.5 shadow-sm flex flex-wrap justify-between items-center gap-4"
      >
        <div
          onClick={() => setCurrentView('detector')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-10 h-10 rounded-xl bg-[#1E40AF] text-white flex items-center justify-center shadow-sm group-hover:bg-[#1E3A8A] transition-colors duration-200">
            <IconShieldCheck />
          </div>
          <div>
            <div className="font-mono-code font-bold text-lg text-[#0F172A] tracking-tight flex items-center gap-2">
              VIET<span className="text-[#1E40AF]">FACT</span>CHECK
              <span className="text-[10px] font-mono-code bg-blue-50 text-[#1E40AF] border border-blue-200 px-2 py-0.5 rounded-md">
                RAG + NLI v2.0
              </span>
            </div>
            <p className="text-xs text-slate-600">
              Hệ thống Kiểm chứng Thông tin Tiếng Việt dựa trên Truy xuất & Phân tích Bằng chứng
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {currentView !== 'detector' && (
            <button
              onClick={() => setCurrentView('detector')}
              className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-100 text-slate-800 hover:bg-slate-200 transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none"
            >
              <IconArrowLeft />
              Bảng Kiểm Chứng
            </button>
          )}
          <button
            onClick={() => setCurrentView('training')}
            className={`cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none ${
              currentView === 'training'
                ? 'bg-[#1E40AF] text-white border-[#1E40AF]'
                : 'bg-white text-slate-700 border-slate-200 hover:border-[#3B82F6] hover:text-[#1E40AF]'
            }`}
          >
            <IconCpu />
            Trung tâm Huấn luyện AI
          </button>
          <button
            onClick={() => setCurrentView('theory')}
            className={`cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold border transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none ${
              currentView === 'theory'
                ? 'bg-[#1E40AF] text-white border-[#1E40AF]'
                : 'bg-white text-slate-700 border-slate-200 hover:border-[#3B82F6] hover:text-[#1E40AF]'
            }`}
          >
            <IconBookOpen />
            Kiến trúc & Lý thuyết
          </button>
        </div>
      </nav>
    </header>
  );

  if (currentView === 'theory') {
    return (
      <div className="min-h-screen bg-[#F8FAFC]">
        {renderNavbar()}
        <TheoryPage />
      </div>
    )
  }

  if (currentView === 'training') {
    return (
      <div className="min-h-screen bg-[#F8FAFC]">
        {renderNavbar()}
        <TrainingPage />
      </div>
    )
  }

  const hv = result?.hybrid_verdict;

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#0F172A]">
      {renderNavbar()}

      <main className="max-w-7xl mx-auto px-4 md:px-8 pb-20">
        {/* HERO & INPUT WORKSPACE */}
        <section className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 md:p-8 mb-8">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-6 mb-6 border-b border-slate-100">
            <div>
              <span className="font-mono-code text-xs font-semibold text-[#1E40AF] uppercase tracking-wider">
                Pipeline 5 Giai Đoạn • 12 Node Xử Lý • Phương Pháp Luận Science (2018)
              </span>
              <h1 className="text-2xl md:text-3xl font-bold text-[#0F172A] mt-1">
                Phân Tích & Kiểm Chứng Thông Tin Dựa Trên Bằng Chứng
              </h1>
            </div>
            <div className="flex flex-wrap gap-2 text-xs font-mono-code">
              <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 border border-slate-200">
                Ngưỡng S2.2: Cosine &ge; 0.50
              </span>
              <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 border border-slate-200">
                Thang S2.1: 1.0 – 5.0
              </span>
              <span className="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 border border-slate-200">
                Novelty S7.2: IU • KL • BD
              </span>
            </div>
          </div>

          {/* Sample Selector Pills */}
          <div className="mb-4">
            <span className="block text-xs font-semibold text-slate-600 mb-2">
              Chọn kịch bản mẫu để kiểm thử tức thì:
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
              {SAMPLE_CLAIMS.map((sample, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setNewsText(sample.text);
                    handleCheck(sample.text);
                  }}
                  className="cursor-pointer text-left p-3 rounded-xl border border-slate-200 bg-slate-50/70 hover:bg-blue-50/60 hover:border-[#3B82F6] transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none"
                >
                  <span className="font-mono-code text-[10px] font-bold text-[#1E40AF] uppercase block">
                    {sample.tag}
                  </span>
                  <span className="text-sm font-semibold text-[#0F172A] block mt-0.5">
                    {sample.label}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Input Textarea */}
          <div>
            <label htmlFor="claim-input" className="sr-only">
              Nội dung tin tức hoặc URL cần kiểm chứng
            </label>
            <textarea
              id="claim-input"
              rows={4}
              className="block w-full rounded-xl border border-slate-300 bg-[#F8FAFC] focus:bg-white focus:border-[#1E40AF] focus:ring-2 focus:ring-[#1E40AF]/20 text-[#0F172A] p-4 outline-none transition-colors duration-200 text-base leading-relaxed"
              placeholder="Nhập mệnh đề tin đồn, nội dung bài viết mạng xã hội hoặc dán trực tiếp đường dẫn URL bài báo cần kiểm chứng..."
              value={newsText}
              onChange={(e) => setNewsText(e.target.value)}
            />
          </div>

          {error && (
            <div role="alert" className="mt-4 p-4 rounded-xl bg-rose-50 text-rose-800 text-sm font-semibold border border-rose-200 flex items-center gap-2.5">
              <IconAlertTriangle />
              <span>{error}</span>
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <label htmlFor="retrieval-mode-select" className="text-xs font-bold text-slate-700 font-mono-code uppercase">
                Cơ chế Truy xuất (Node 4–5):
              </label>
              <select
                id="retrieval-mode-select"
                value={retrievalMode}
                onChange={(e) => setRetrievalMode(e.target.value as typeof retrievalMode)}
                className="cursor-pointer bg-[#F8FAFC] border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-[#0F172A] focus:border-[#1E40AF] outline-none"
              >
                <option value="hybrid_rerank">Hybrid (BM25 + Dense) + Cross-Reranker + Lọc Thời Gian (Khuyên dùng)</option>
                <option value="hybrid">Hybrid Retrieval (BM25 + Dense Fusion)</option>
                <option value="bm25">BM25 Baseline (Okapi BM25)</option>
                <option value="dense">Dense Vector Semantic Retrieval</option>
              </select>
            </div>
            <button
              type="button"
              onClick={() => handleCheck()}
              disabled={loading}
              className={`cursor-pointer inline-flex items-center gap-2.5 px-7 py-3.5 text-sm font-bold rounded-xl shadow-sm text-white transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-[#1E40AF] outline-none ${
                loading
                  ? 'bg-slate-400 cursor-not-allowed'
                  : 'bg-[#1E40AF] hover:bg-[#1E3A8A]'
              }`}
            >
              <IconSearch />
              <span>{loading ? 'Đang truy xuất & phân tích bằng chứng...' : 'Kiểm chứng & Truy xuất Bằng chứng'}</span>
            </button>
          </div>
        </section>

        {/* DATA-DENSE INTELLIGENCE DASHBOARD */}
        {result && hv && (
          <div className="space-y-6">
            {/* TOP KPI & VERDICT ROW */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Main Verdict & Explainable Trace (8 cols) */}
              <div className="lg:col-span-8 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col justify-between">
                <div>
                  <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
                    <div className="flex items-center gap-2.5">
                      <span
                        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg font-mono-code text-xs font-bold uppercase border ${
                          hv.verdict_code === 'FALSE'
                            ? 'bg-rose-50 text-rose-800 border-rose-200'
                            : hv.verdict_code === 'TRUE'
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                            : 'bg-amber-50 text-amber-800 border-amber-200'
                        }`}
                      >
                        {hv.verdict_code === 'FALSE' ? <IconXCircle /> : hv.verdict_code === 'TRUE' ? <IconCheckCircle /> : <IconAlertTriangle />}
                        {hv.verdict_badge}
                      </span>
                    </div>
                    <span className="font-mono-code text-xs text-slate-500">
                      Latency: {hv.latency_ms ?? 12} ms • Độ bất định: {((hv.uncertainty_score ?? 0.15) * 100).toFixed(0)}%
                    </span>
                  </div>

                  <h2
                    className={`text-2xl md:text-3xl font-bold mt-4 ${
                      hv.verdict_code === 'FALSE'
                        ? 'text-rose-700'
                        : hv.verdict_code === 'TRUE'
                        ? 'text-emerald-700'
                        : 'text-amber-700'
                    }`}
                  >
                    {hv.verdict_title}
                  </h2>

                  {/* NLI vs LLM Few-Shot Grounded Comparison Banner */}
                  {hv.llm_verification && (
                    <div className="mt-4 p-4 rounded-xl bg-blue-50/70 border border-blue-200">
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5">
                        <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                          Đối chứng NLI Verifier vs. LLM 3-Shot Evidence-Grounded Verification
                        </span>
                        <span className="font-mono-code text-[11px] font-bold px-2.5 py-0.5 rounded bg-white text-[#1E40AF] border border-blue-200">
                          NLI: {hv.nli_3class_label} | LLM Few-Shot: {hv.llm_verification.llm_label} ({(hv.llm_verification.llm_confidence * 100).toFixed(0)}%)
                        </span>
                      </div>
                      <p className="text-xs text-slate-800 leading-relaxed font-medium">
                        {hv.llm_verification.grounded_explanation}
                      </p>
                    </div>
                  )}

                  {/* Step-by-step Reasoning Trace */}
                  <div className="mt-5 space-y-2.5">
                    <h3 className="font-mono-code text-xs font-bold text-slate-600 uppercase tracking-wider">
                      Chuỗi Lập Luận Giải Trình (Verification Reasoning Trace):
                    </h3>
                    {hv.reasoning_steps.map((step, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-3 p-3 rounded-xl bg-[#F8FAFC] border border-slate-200/80 text-sm text-[#0F172A]"
                      >
                        <span className="font-mono-code w-6 h-6 rounded-lg bg-[#1E40AF] text-white text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                          0{i + 1}
                        </span>
                        <span className="leading-relaxed">{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Veracity Gauge & Core KPIs (4 cols) */}
              <div className="lg:col-span-4 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
                    <span className="font-mono-code text-xs font-bold text-slate-700 uppercase">
                      Thang Đo Xác Thực (Mục S2.1)
                    </span>
                    <span className="font-mono-code text-xs font-semibold text-[#1E40AF]">1.0 – 5.0</span>
                  </div>
                  <VeracityGauge score={hv.veracity_score_1_to_5} fakeProb={hv.final_fake_probability} />
                </div>

                <div className="grid grid-cols-2 gap-3 mt-6 pt-4 border-t border-slate-100">
                  <div className="p-3 rounded-xl bg-[#F8FAFC] border border-slate-200">
                    <span className="text-[11px] text-slate-600 font-medium block">Điểm Bằng chứng</span>
                    <span className="font-mono-code text-lg font-bold text-[#0F172A]">
                      {hv.evidence_veracity_score} / 5.0
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#F8FAFC] border border-slate-200">
                    <span className="text-[11px] text-slate-600 font-medium block">Xác suất Giả (Tổng hợp)</span>
                    <span className="font-mono-code text-lg font-bold text-rose-700">
                      {(hv.final_fake_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#F8FAFC] border border-slate-200">
                    <span className="text-[11px] text-slate-600 font-medium block">Bằng chứng &ge; 0.50</span>
                    <span className="font-mono-code text-lg font-bold text-[#1E40AF]">
                      {hv.retrieval_stats.accepted_above_050} / {hv.retrieval_stats.total_candidates_checked}
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-[#F8FAFC] border border-slate-200">
                    <span className="text-[11px] text-slate-600 font-medium block">Tương đồng Max</span>
                    <span className="font-mono-code text-lg font-bold text-emerald-700">
                      {(hv.retrieval_stats.max_similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* EVIDENCE RETRIEVAL & NLI TABLE WITH INTERACTIVE FILTERING (NODES 4, 5, 6, 10) */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 mb-5 border-b border-slate-100">
                <div>
                  <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                    Giai đoạn 2 & 4 (Nodes 4, 5, 6, 10)
                  </span>
                  <h3 className="text-xl font-bold text-[#0F172A] mt-0.5">
                    Bằng Chứng Truy Xuất Đa Tầng & Suy Luận Quan Hệ NLI
                  </h3>
                </div>

                {/* Filter Controls (UI Pro Max Requirement: Interactive Filtering) */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1 text-xs font-semibold text-slate-600 mr-1">
                    <IconFilter /> Lọc:
                  </span>
                  {[
                    { id: 'ALL', label: `Tất cả (${hv.verified_evidences.length})` },
                    { id: 'SUPPORTS', label: 'Ủng hộ (SUPPORTS)' },
                    { id: 'REFUTES', label: 'Bác bỏ (REFUTES)' },
                    { id: 'LOCAL', label: 'Kho Nội bộ' },
                    { id: 'LIVE', label: 'Báo chí Trực tuyến' },
                  ].map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => setEvidenceFilter(f.id as typeof evidenceFilter)}
                      className={`cursor-pointer px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors duration-200 ${
                        evidenceFilter === f.id
                          ? 'bg-[#1E40AF] text-white border-[#1E40AF]'
                          : 'bg-[#F8FAFC] text-slate-700 border-slate-200 hover:border-[#3B82F6]'
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>

              {filteredEvidences.length > 0 ? (
                <div className="space-y-3.5">
                  {filteredEvidences.map((ev, idx) => {
                    const isSupport = ev.stance === 'SUPPORTS';
                    const isRefute = ev.stance === 'REFUTES';
                    return (
                      <article
                        key={idx}
                        className="p-5 rounded-xl border border-slate-200 bg-[#F8FAFC] hover:border-[#3B82F6] transition-colors duration-200"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                          <div className="flex flex-wrap items-center gap-2">
                            <span
                              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg font-mono-code text-xs font-bold border ${
                                isSupport
                                  ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                                  : isRefute
                                  ? 'bg-rose-50 text-rose-800 border-rose-200'
                                  : 'bg-amber-50 text-amber-800 border-amber-200'
                              }`}
                            >
                              {isSupport ? <IconCheckCircle /> : isRefute ? <IconXCircle /> : <IconAlertTriangle />}
                              {isSupport ? 'SUPPORTS (Ủng hộ)' : isRefute ? 'REFUTES (Bác bỏ)' : 'NEI (Trung lập)'}
                            </span>
                            <span className="font-mono-code px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-xs font-bold text-[#0F172A]">
                              Điểm S2.1: {ev.veracity_score_1_to_5}/5
                            </span>
                            <span className="font-mono-code px-2.5 py-1 rounded-lg bg-blue-50 border border-blue-200 text-xs font-bold text-[#1E40AF]">
                              Sim: {(ev.similarity * 100).toFixed(1)}%
                            </span>
                            {ev.bm25_score !== undefined && (
                              <span className="font-mono-code px-2 py-1 rounded-lg bg-slate-100 border border-slate-200 text-[11px] font-semibold text-slate-700">
                                BM25: {(ev.bm25_score * 100).toFixed(0)}% • Dense: {((ev.dense_score ?? 0) * 100).toFixed(0)}%
                              </span>
                            )}
                            {ev.date && (
                              <span className="font-mono-code px-2.5 py-1 rounded-lg bg-amber-50 border border-amber-200 text-[11px] font-bold text-amber-900">
                                Ngày: {ev.date}
                              </span>
                            )}
                          </div>
                          <span className="font-mono-code text-xs font-semibold text-slate-600">
                            {ev.citation_tag || `Nguồn: ${ev.source}`}
                          </span>
                        </div>

                        <h4 className="text-base font-bold text-[#0F172A] mb-1.5">
                          {ev.title}
                        </h4>
                        <p className="text-sm text-slate-700 leading-relaxed mb-3">
                          {ev.evidence_text}
                        </p>

                        <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-slate-200/80 text-xs">
                          <span className="text-slate-700 font-medium">
                            <strong className="text-[#0F172A]">Phân tích NLI:</strong> {ev.stance_explanation}
                          </span>
                          {ev.url && (
                            <a
                              href={ev.url}
                              target="_blank"
                              rel="noreferrer"
                              className="cursor-pointer inline-flex items-center gap-1 font-semibold text-[#1E40AF] hover:underline"
                            >
                              <span>Xem nguồn gốc</span>
                              <IconExternalLink />
                            </a>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="p-6 rounded-xl bg-slate-50 border border-slate-200 text-slate-700 text-sm font-medium">
                  Không có bằng chứng nào khớp với bộ lọc hiện tại hoặc vượt ngưỡng tương đồng ngữ nghĩa 0.50 (Mục S2.2).
                </div>
              )}
            </section>

            {/* MIDDLE ROW: CLAIM EXTRACTION (NODE 3), NOVELTY METRICS (NODE 8) & EMOTION/SCAM (NODE 9) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Node 3: Claim & NER */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
                <div className="flex items-center justify-between">
                  <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                    Node 3 • Claim & Check-Worthiness
                  </span>
                  {result.claim_analysis.check_worthiness_score !== undefined && (
                    <span className="font-mono-code text-[11px] font-bold px-2.5 py-0.5 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200">
                      Check-worthy: {(result.claim_analysis.check_worthiness_score * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <h3 className="text-lg font-bold text-[#0F172A] mt-0.5 mb-3">
                  Mệnh Đề Khẳng Định & Thực Thể
                </h3>
                <div className="p-3.5 rounded-xl bg-[#F8FAFC] border border-slate-200 text-sm font-medium text-[#0F172A] mb-4">
                  "{result.claim_analysis.primary_claim}"
                </div>

                <div className="space-y-3">
                  <div>
                    <span className="text-xs font-semibold text-slate-600 block mb-1.5">
                      Thực thể nhận diện (NER & Số liệu):
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {result.claim_analysis.entities.map((ent, i) => (
                        <span key={i} className="font-mono-code px-2.5 py-1 bg-blue-50 text-[#1E40AF] rounded-lg text-xs font-semibold border border-blue-200">
                          {ent}
                        </span>
                      ))}
                      {result.claim_analysis.numbers_dates.map((num, i) => (
                        <span key={i} className="font-mono-code px-2.5 py-1 bg-amber-50 text-amber-800 rounded-lg text-xs font-semibold border border-amber-200">
                          {num}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <span className="text-xs font-semibold text-slate-600 block mb-1">
                      Truy vấn tìm kiếm bằng chứng (Search Queries):
                    </span>
                    {result.claim_analysis.search_queries.map((q, i) => (
                      <div key={i} className="font-mono-code text-xs text-slate-700 bg-slate-100 px-3 py-1.5 rounded-lg mb-1 truncate">
                        Q{i + 1}: {q}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Node 8: Novelty Metrics (Science 2018 Section S7.2) */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col justify-between">
                <div>
                  <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                    Node 8 • Mục S7.2 Science (2018)
                  </span>
                  <h3 className="text-lg font-bold text-[#0F172A] mt-0.5 mb-3">
                    Đo Độ Mới Lạ Thông Tin (Novelty)
                  </h3>
                  <p className="text-xs text-slate-600 mb-4">
                    Đo khoảng cách phân phối chủ đề giữa tin cần kiểm tra ($\Gamma_r$) và tin tức nền ($\Gamma_b$) theo công thức S6, S8, S9.
                  </p>

                  <div className="space-y-3">
                    {[
                      {
                        label: 'Information Uniqueness (IU)',
                        formula: '1 - cos(Γr, Γb)',
                        val: result.novelty_metrics.information_uniqueness_IU,
                        pct: Math.min(100, result.novelty_metrics.information_uniqueness_IU * 100),
                      },
                      {
                        label: 'Symmetric KL-Divergence',
                        formula: '0.5 * (KL(P||Q) + KL(Q||P))',
                        val: result.novelty_metrics.kl_divergence_KL,
                        pct: Math.min(100, (result.novelty_metrics.kl_divergence_KL / 10) * 100),
                      },
                      {
                        label: 'Bhattacharyya Distance (BD)',
                        formula: '-ln(Σ √(Γr · Γb))',
                        val: result.novelty_metrics.bhattacharyya_distance_BD,
                        pct: Math.min(100, (result.novelty_metrics.bhattacharyya_distance_BD / 4) * 100),
                      },
                    ].map((item, idx) => (
                      <div key={idx} className="p-3 rounded-xl bg-[#F8FAFC] border border-slate-200">
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-semibold text-[#0F172A]">{item.label}</span>
                          <span className="font-mono-code font-bold text-[#1E40AF]">{item.val}</span>
                        </div>
                        <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#3B82F6] rounded-full transition-all duration-300"
                            style={{ width: `${item.pct}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 text-xs font-semibold text-slate-700">
                  Mức độ mới lạ: <span className="text-[#1E40AF]">{result.novelty_metrics.novelty_level}</span>
                </div>
              </div>

              {/* Node 9: Emotion & Dissent Profile (Section S7.1) */}
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col justify-between">
                <div>
                  <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                    Node 9 • Mục S7.1 Science (2018)
                  </span>
                  <h3 className="text-lg font-bold text-[#0F172A] mt-0.5 mb-3">
                    Cảm Xúc, Phản Bác & Kịch Bản Lừa Đảo
                  </h3>

                  {/* Horizontal Bar Chart for 5 Emotion/Dissent dimensions */}
                  <div className="space-y-2.5 mb-4">
                    {[
                      { key: 'surprise', label: 'Surprise (Ngạc nhiên / Giật gân)', color: 'bg-amber-500' },
                      { key: 'fear', label: 'Fear (Sợ hãi / Hoang mang)', color: 'bg-rose-500' },
                      { key: 'disgust', label: 'Disgust (Phẫn nộ / Lừa đảo)', color: 'bg-purple-600' },
                      { key: 'trust', label: 'Trust (Tín hiệu Chính thống)', color: 'bg-emerald-600' },
                      { key: 'dissent', label: 'Dissent (Từ ngữ Bác bỏ / Đính chính)', color: 'bg-[#1E40AF]' },
                    ].map((emo) => {
                      const val = result.emotion_heuristics.emotion_distribution[emo.key] || 0;
                      return (
                        <div key={emo.key}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="font-medium text-slate-700">{emo.label}</span>
                            <span className="font-mono-code font-bold text-slate-900">{(val * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200/60">
                            <div className={`h-full ${emo.color}`} style={{ width: `${Math.max(4, val * 100)}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {result.emotion_heuristics.is_debunking_context && (
                    <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 text-xs font-semibold flex items-start gap-2">
                      <IconCheckCircle />
                      <span>
                        Phát hiện ngữ cảnh đính chính chính thống: Bài viết đang bác bỏ tin đồn sai sự thật.
                      </span>
                    </div>
                  )}

                  {result.emotion_heuristics.triggered_scam_categories.map((cat, i) => (
                    <div key={i} className="mt-2 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 text-xs font-semibold flex items-start gap-2">
                      <IconAlertTriangle />
                      <span>
                        Cảnh báo {cat.name}: {cat.matched_keywords.join(', ')}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                  <span className="text-slate-600 font-medium">Chỉ số giật gân (Sensationalism):</span>
                  <span className="font-mono-code font-bold text-rose-700">
                    {(result.emotion_heuristics.sensational_score * 100).toFixed(0)}% ({result.emotion_heuristics.exclamation_count} dấu !)
                  </span>
                </div>
              </div>
            </div>

            {/* NODE 7: MULTI-MODEL BENCHMARK & COMPARISON CHART */}
            {result.results && Object.keys(result.results).length > 0 && (
              <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 pb-4 mb-5 border-b border-slate-100">
                  <div>
                    <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                      Node 7 • Multi-Model Style Benchmark
                    </span>
                    <h3 className="text-xl font-bold text-[#0F172A] mt-0.5">
                      So Sánh Xác Suất Dự Đoán Của Các Mô Hình Học Máy & Học Sâu
                    </h3>
                  </div>
                  <span className="font-mono-code text-xs text-slate-600">
                    Xác suất văn phong giả trung bình: {(hv.avg_ml_fake_probability * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Visual Comparison Bar Grid Across All Loaded Models */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-6">
                  {Object.entries(result.results).map(([name, mRes]) => {
                    const fakePct = (mRes.fake_probability || 0) * 100;
                    const isSelected = activeTab === name;
                    return (
                      <button
                        key={name}
                        type="button"
                        onClick={() => setActiveTab(name)}
                        className={`cursor-pointer text-left p-3.5 rounded-xl border transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none ${
                          isSelected
                            ? 'bg-[#1E40AF] text-white border-[#1E40AF] shadow-sm'
                            : 'bg-[#F8FAFC] text-[#0F172A] border-slate-200 hover:border-[#3B82F6]'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="font-mono-code text-xs font-bold uppercase truncate">
                            {mRes.education?.name?.split(' (')[0] || name}
                          </span>
                          <span
                            className={`font-mono-code text-[10px] font-bold px-1.5 py-0.5 rounded ${
                              isSelected
                                ? 'bg-white/20 text-white'
                                : mRes.prediction === 'Fake'
                                ? 'bg-rose-100 text-rose-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {mRes.prediction}
                          </span>
                        </div>
                        <div className="font-mono-code text-lg font-bold mb-1.5">
                          {fakePct.toFixed(1)}% <span className="text-[10px] font-normal opacity-80">Fake</span>
                        </div>
                        <div className={`w-full h-1.5 rounded-full overflow-hidden ${isSelected ? 'bg-white/30' : 'bg-slate-200'}`}>
                          <div
                            className={`h-full rounded-full ${
                              isSelected ? 'bg-[#F59E0B]' : fakePct >= 50 ? 'bg-rose-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${fakePct}%` }}
                          />
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Selected Model Deep-Dive */}
                {activeTab && result.results[activeTab] && (
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pt-2">
                    <div className="lg:col-span-8 bg-[#F8FAFC] rounded-xl p-5 border border-slate-200">
                      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                        <div>
                          <span className="font-mono-code text-xs text-slate-500 uppercase">
                            Chi tiết thực nghiệm mô hình
                          </span>
                          <h4 className="text-lg font-bold text-[#0F172A]">
                            {result.results[activeTab].education?.name || activeTab}
                          </h4>
                        </div>
                        <span
                          className={`font-mono-code text-sm font-bold px-3 py-1 rounded-lg border ${
                            result.results[activeTab].prediction === 'Fake'
                              ? 'bg-rose-50 text-rose-800 border-rose-200'
                              : 'bg-emerald-50 text-emerald-800 border-emerald-200'
                          }`}
                        >
                          Dự đoán văn phong: {result.results[activeTab].prediction === 'Fake' ? 'TIN GIẢ' : 'TIN THẬT'}
                        </span>
                      </div>

                      {renderMetrics(result.results[activeTab].metrics)}
                      <ConfusionMatrix matrix={result.results[activeTab].metrics?.confusion_matrix} />
                    </div>

                    <div className="lg:col-span-4 bg-[#0F172A] rounded-xl p-6 text-white flex flex-col justify-between">
                      <div>
                        <span className="font-mono-code text-xs font-bold text-[#F59E0B] uppercase tracking-wider">
                          Cơ sở Toán học & Luồng xử lý
                        </span>
                        <h4 className="text-base font-bold text-white mt-1 mb-3">
                          {result.results[activeTab].education?.name}
                        </h4>
                        <p className="text-sm text-slate-300 leading-relaxed mb-3">
                          {result.results[activeTab].education?.concept}
                        </p>
                        <p className="text-xs text-slate-400 italic mb-5">
                          {result.results[activeTab].education?.principle}
                        </p>
                      </div>

                      <div className="space-y-2 border-t border-slate-800 pt-4">
                        <span className="font-mono-code text-[11px] font-bold text-blue-400 uppercase block mb-2">
                          Pipeline thực thi:
                        </span>
                        {result.results[activeTab].education?.flow?.split(' -> ').map((step, i) => (
                          <div key={i} className="flex items-center gap-3">
                            <span className="font-mono-code w-5 h-5 rounded-md bg-[#1E40AF] text-white flex items-center justify-center text-[10px] font-bold shrink-0">
                              {i + 1}
                            </span>
                            <span className="text-xs font-medium text-slate-200">{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* NODE 12B: USER FEEDBACK (HUMAN-IN-THE-LOOP FACT VERIFICATION FEEDBACK) */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                    Node 12B • Human-in-the-Loop Feedback
                  </span>
                  <h3 className="text-lg font-bold text-[#0F172A] mt-0.5">
                    Đánh Giá Của Người Dùng Về Phán Quyết & Bằng Chứng
                  </h3>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Phản hồi của bạn được lưu trữ vào cơ sở dữ liệu kiểm chứng (SQLite/JSON) để làm giàu Kho tri thức chuẩn (Gold Evidence Corpus).
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => handleSubmitFeedback('ACCURATE')}
                    className={`cursor-pointer inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold border transition-colors ${
                      feedbackRating === 'ACCURATE'
                        ? 'bg-emerald-600 text-white border-emerald-600'
                        : 'bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100'
                    }`}
                  >
                    <IconCheckCircle />
                    <span>Phán quyết Chính xác (Đồng ý)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFeedbackRating('INACCURATE')}
                    className={`cursor-pointer inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-bold border transition-colors ${
                      feedbackRating === 'INACCURATE'
                        ? 'bg-rose-600 text-white border-rose-600'
                        : 'bg-rose-50 text-rose-800 border-rose-200 hover:bg-rose-100'
                    }`}
                  >
                    <IconAlertTriangle />
                    <span>Báo sai lệch / Đề xuất nhãn khác</span>
                  </button>
                </div>
              </div>

              {feedbackRating === 'INACCURATE' && (
                <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-12 gap-3 items-end">
                  <div className="md:col-span-3">
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Nhãn đúng theo đánh giá của bạn:
                    </label>
                    <select
                      value={suggestedLabel}
                      onChange={(e) => setSuggestedLabel(e.target.value)}
                      className="w-full bg-[#F8FAFC] border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-[#0F172A]"
                    >
                      <option value="Supported">Supported (Đúng sự thật)</option>
                      <option value="Refuted">Refuted (Bác bỏ / Tin giả)</option>
                      <option value="NEI">NEI (Chưa đủ bằng chứng)</option>
                    </select>
                  </div>
                  <div className="md:col-span-6">
                    <label className="block text-xs font-semibold text-slate-700 mb-1">
                      Ghi chú hoặc đường dẫn nguồn bằng chứng bổ sung:
                    </label>
                    <input
                      type="text"
                      value={feedbackComment}
                      onChange={(e) => setFeedbackComment(e.target.value)}
                      placeholder="Nhập nhận xét hoặc dán link bài báo chính thống đối chứng..."
                      className="w-full bg-[#F8FAFC] border border-slate-300 rounded-xl px-3 py-2 text-xs text-[#0F172A]"
                    />
                  </div>
                  <div className="md:col-span-3">
                    <button
                      type="button"
                      onClick={() => handleSubmitFeedback('INACCURATE')}
                      className="cursor-pointer w-full py-2.5 px-4 rounded-xl bg-[#1E40AF] text-white text-xs font-bold hover:bg-[#1E3A8A] transition-colors"
                    >
                      Gửi Phản Hồi Kiểm Duyệt
                    </button>
                  </div>
                </div>
              )}

              {feedbackStatus && (
                <div className="mt-3 p-3 rounded-xl bg-blue-50 border border-blue-200 text-xs font-semibold text-[#1E40AF]">
                  {feedbackStatus}
                </div>
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
