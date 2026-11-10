import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

interface SystemStats {
  cpu: number;
  ram_total: number;
  ram_used: number;
  ram_percent: number;
  disk: number;
  gpus: Array<{
    name: string;
    usage: number;
    memory_total: number;
    memory_used: number;
  }>;
}

interface TrainingDetails {
  current: number;
  total: number;
  elapsed: string;
  remaining: string;
  speed: string;
}

interface TrainingResult {
  accuracy: number;
  f1: number;
  precision: number;
  recall: number;
  false_alarm_rate: number;
  duration: number;
  confusion_matrix: number[][];
}

interface BenchmarkMetrics {
  dataset_info?: {
    name: string;
    total_claims: number;
    corpus_size: number;
  };
  official_metrics_summary?: {
    claim_detection_macro_f1: number;
    retrieval_recall_at_5: number;
    retrieval_mrr_at_5: number;
    retrieval_ndcg_at_5: number;
    verification_macro_f1: number;
    evidence_precision: number;
    evidence_recall: number;
    citation_accuracy: number;
    end_to_end_evidence_grounded_accuracy: number;
    avg_latency_ms: number;
  };
  retrieval_comparison?: Record<string, {
    method_name: string;
    recall_at_1: number;
    recall_at_3: number;
    recall_at_5: number;
    mrr_at_5: number;
    ndcg_at_5: number;
    evidence_precision: number;
    avg_latency_ms: number;
  }>;
  verification_comparison?: Record<string, {
    name: string;
    macro_f1: number;
    accuracy: number;
    precision: number;
    recall: number;
  }>;
}

const IconPlay = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
);

const IconStop = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <rect x="4" y="4" width="16" height="16" rx="2" />
  </svg>
);

const IconAlert = () => (
  <svg className="w-5 h-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const TrainingPage: React.FC = () => {
  const [modelType, setModelType] = useState('all');
  const [epochs, setEpochs] = useState(10);
  const [isTraining, setIsTraining] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [details, setDetails] = useState<TrainingDetails>({
    current: 0,
    total: 0,
    elapsed: '00:00',
    remaining: '00:00',
    speed: '0it/s'
  });
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [results, setResults] = useState<Record<string, TrainingResult>>({});
  const [benchmark, setBenchmark] = useState<BenchmarkMetrics | null>(null);
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [connectionError, setConnectionError] = useState(false);

  const terminalRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    const fetchStatsAndResults = async () => {
      try {
        const benchRes = await axios.get('http://localhost:8000/benchmark-metrics');
        if (benchRes.data) setBenchmark(benchRes.data);
      } catch {
        // ignore if port 8000 not queried yet
      }
      try {
        const [statsRes, resultsRes] = await Promise.all([
          axios.get('http://localhost:8001/system-stats'),
          axios.get('http://localhost:8001/training-results'),
        ]);
        setSystemStats(statsRes.data);
        if (resultsRes.data) setResults(resultsRes.data);
        setConnectionError(false);
      } catch {
        setConnectionError(true);
      }
    };
    fetchStatsAndResults();
    const interval = setInterval(fetchStatsAndResults, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const recoverState = async () => {
      try {
        const response = await axios.get('http://localhost:8001/training-status');
        const data = response.data;
        if (data.is_training) {
          setLogs(data.logs);
          setProgress(data.progress);
          setDetails(data.details);
          setModelType(data.current_model);
          setIsTraining(true);
          setStatus('running');
          connectSSE();
        } else if (data.logs.length > 0) {
          setLogs(data.logs);
          setProgress(data.progress);
          setDetails(data.details);
          setStatus(data.progress === 100 ? 'completed' : 'idle');
        }
      } catch {
        // ignore on initial mount if port 8001 is offline
      }
    };
    recoverState();
  }, []);

  const connectSSE = () => {
    if (eventSourceRef.current) eventSourceRef.current.close();

    const eventSource = new EventSource('http://localhost:8001/training-events');
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'log') {
        setLogs((prev) => {
          if (prev.includes(data.content) && prev.indexOf(data.content) > prev.length - 5) return prev;
          return [...prev, data.content];
        });
        if (data.progress !== undefined) setProgress(data.progress);
        if (data.details) setDetails(data.details);
      } else if (data.type === 'status') {
        if (data.status === 'completed') {
          setProgress(100);
          setIsTraining(false);
          setStatus('completed');
          eventSource.close();
        } else if (data.status === 'failed') {
          setIsTraining(false);
          setStatus('failed');
          eventSource.close();
        }
      }
    };
  };

  const startTraining = async () => {
    setLogs(['[SYSTEM] Khởi tạo tiến trình huấn luyện mô hình AI...']);
    setProgress(0);
    setIsTraining(true);
    setStatus('running');

    try {
      await axios.post('http://localhost:8001/start-training', {
        model_type: modelType,
        epochs: epochs
      });
      connectSSE();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setLogs((prev) => [...prev, `[ERROR] Không thể khởi động: ${msg}`]);
      setIsTraining(false);
      setStatus('failed');
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('http://localhost:8001/stop-training');
      setIsTraining(false);
      setStatus('idle');
    } catch (err) {
      console.error('Failed to stop training', err);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-8 pb-20">
      {connectionError && (
        <div
          role="alert"
          className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 font-semibold text-sm flex items-center gap-3"
        >
          <IconAlert />
          <span>
            Máy chủ huấn luyện (Port 8001) chưa bật. Chạy lệnh <code className="font-mono-code bg-rose-100 px-1.5 py-0.5 rounded">python training_api.py</code> hoặc <code className="font-mono-code bg-rose-100 px-1.5 py-0.5 rounded">run_all.bat</code> để điều khiển huấn luyện trực tiếp.
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Config & Hardware Telemetry (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
            <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
              Training Controller
            </span>
            <h2 className="text-lg font-bold text-[#0F172A] mt-0.5 mb-5">
              Cấu Hình Huấn Luyện Mô Hình
            </h2>

            <div className="space-y-4">
              <div>
                <label htmlFor="model-select" className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Chọn nhóm mô hình huấn luyện
                </label>
                <select
                  id="model-select"
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  disabled={isTraining}
                  className="cursor-pointer w-full bg-[#F8FAFC] border border-slate-300 rounded-xl px-3.5 py-2.5 text-sm font-semibold text-[#0F172A] focus:border-[#1E40AF] outline-none"
                >
                  <option value="all">5 Mô hình ML + Doc2Vec + Full Benchmark</option>
                  <option value="benchmark">Chạy Đánh Giá Benchmark Đề Tài 8 (VNFC-360)</option>
                  <option value="phobert">VinAI PhoBERT (Transformer)</option>
                  <option value="vibert">FPTAI ViBERT (Transformer)</option>
                  <option value="sbert">Vietnamese SBERT (Bi-Encoder)</option>
                </select>
              </div>

              <div>
                <label htmlFor="epoch-input" className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Số vòng lặp tối đa (Epochs)
                </label>
                <input
                  id="epoch-input"
                  type="number"
                  min={1}
                  max={50}
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                  disabled={isTraining}
                  className="font-mono-code w-full bg-[#F8FAFC] border border-slate-300 rounded-xl px-3.5 py-2.5 text-sm font-bold text-[#0F172A] focus:border-[#1E40AF] outline-none"
                />
              </div>

              {!isTraining ? (
                <button
                  type="button"
                  onClick={startTraining}
                  className="cursor-pointer w-full py-3.5 rounded-xl font-bold text-sm transition-colors duration-200 bg-[#1E40AF] text-white hover:bg-[#1E3A8A] shadow-sm flex items-center justify-center gap-2 focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none"
                >
                  <IconPlay />
                  <span>Bắt đầu Huấn luyện</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopTraining}
                  className="cursor-pointer w-full py-3.5 rounded-xl font-bold text-sm transition-colors duration-200 bg-rose-600 text-white hover:bg-rose-700 shadow-sm flex items-center justify-center gap-2 focus-visible:ring-2 focus-visible:ring-rose-600 outline-none"
                >
                  <IconStop />
                  <span>Dừng Huấn luyện</span>
                </button>
              )}
            </div>
          </div>

          {/* Hardware Telemetry */}
          <div className="bg-[#0F172A] rounded-2xl p-6 text-white shadow-sm">
            <span className="font-mono-code text-xs font-bold text-[#F59E0B] uppercase">
              Hardware Telemetry
            </span>
            <h2 className="text-base font-bold mt-0.5 mb-5">Tài Nguyên Hệ Thống</h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between font-mono-code text-xs mb-1.5 text-slate-300">
                  <span>CPU Load</span>
                  <span>{systemStats?.cpu ?? 0}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#3B82F6] transition-all duration-300"
                    style={{ width: `${systemStats?.cpu || 0}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between font-mono-code text-xs mb-1.5 text-slate-300">
                  <span>RAM Usage</span>
                  <span>{systemStats?.ram_percent ?? 0}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#F59E0B] transition-all duration-300"
                    style={{ width: `${systemStats?.ram_percent || 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Terminal & Benchmark Results (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Status KPI Row */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <span className="text-xs font-medium text-slate-500 block">Trạng thái</span>
              <span className="font-mono-code text-base font-bold text-[#1E40AF] uppercase mt-0.5 block">
                {status}
              </span>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <span className="text-xs font-medium text-slate-500 block">Mô hình</span>
              <span className="font-mono-code text-base font-bold text-[#0F172A] uppercase mt-0.5 block">
                {modelType}
              </span>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <span className="text-xs font-medium text-slate-500 block">Tiến độ</span>
              <span className="font-mono-code text-base font-bold text-emerald-700 uppercase mt-0.5 block">
                {progress}%
              </span>
            </div>
          </div>

          {isTraining && details.total > 0 && (
            <div className="bg-[#0F172A] rounded-xl p-4 flex flex-wrap justify-between items-center gap-4 text-white border border-slate-800">
              <div className="flex items-center gap-6 font-mono-code text-xs">
                <div>
                  <span className="text-slate-400 block text-[10px]">ITERATIONS</span>
                  <span className="text-blue-400 font-bold">{details.current} / {details.total}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">ELAPSED / ETA</span>
                  <span className="text-amber-400 font-bold">{details.elapsed} &lt; {details.remaining}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">THROUGHPUT</span>
                  <span className="text-emerald-400 font-bold">{details.speed}</span>
                </div>
              </div>
            </div>
          )}

          {/* Terminal Window */}
          <div className="bg-[#0F172A] rounded-2xl shadow-sm overflow-hidden border border-slate-800">
            <div className="bg-slate-900 px-5 py-3 font-mono-code text-xs font-bold text-slate-300 flex items-center justify-between border-b border-slate-800">
              <span>TRAINING STREAM CONSOLE</span>
              <span className="text-[11px] text-slate-400">SSE Stream (Port 8001)</span>
            </div>
            <div
              ref={terminalRef}
              className="h-[320px] p-5 font-mono-code text-xs text-slate-200 overflow-y-auto space-y-1.5 bg-[#020617]"
            >
              {logs.length > 0 ? (
                logs.map((log, i) => <div key={i}>{log}</div>)
              ) : (
                <div className="text-slate-500">Sẵn sàng nhận lệnh huấn luyện...</div>
              )}
            </div>
            <div className="p-3 bg-slate-900 border-t border-slate-800">
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#3B82F6] rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </div>

          {/* Official Fact-Checking & Retrieval Benchmark Tables (Row 8 Excel Spec) */}
          {benchmark && benchmark.official_metrics_summary && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
              <div>
                <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                  Vietnamese Fact Verification Evaluation Set ({benchmark.dataset_info?.total_claims || 360} Claims)
                </span>
                <h2 className="text-lg font-bold text-[#0F172A] mt-0.5">
                  Bộ Chỉ Số Thực Nghiệm Chuẩn Đề Tài Số 8 (Retrieval & Evidence-Grounded NLI)
                </h2>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: 'Claim Detection F1', val: `${(benchmark.official_metrics_summary.claim_detection_macro_f1 * 100).toFixed(1)}%` },
                  { label: 'Retrieval Recall@5', val: `${(benchmark.official_metrics_summary.retrieval_recall_at_5 * 100).toFixed(1)}%` },
                  { label: 'MRR@5 / NDCG@5', val: `${(benchmark.official_metrics_summary.retrieval_mrr_at_5 * 100).toFixed(1)}% / ${(benchmark.official_metrics_summary.retrieval_ndcg_at_5 * 100).toFixed(1)}%` },
                  { label: 'Verification Macro-F1', val: `${(benchmark.official_metrics_summary.verification_macro_f1 * 100).toFixed(1)}%` },
                  { label: 'Evidence Precision', val: `${(benchmark.official_metrics_summary.evidence_precision * 100).toFixed(1)}%` },
                  { label: 'Citation Accuracy', val: `${(benchmark.official_metrics_summary.citation_accuracy * 100).toFixed(1)}%` },
                  { label: 'E2E Grounded Acc', val: `${(benchmark.official_metrics_summary.end_to_end_evidence_grounded_accuracy * 100).toFixed(1)}%` },
                  { label: 'Avg Latency', val: `${benchmark.official_metrics_summary.avg_latency_ms} ms` },
                ].map((m, idx) => (
                  <div key={idx} className="p-3 rounded-xl bg-[#F8FAFC] border border-slate-200">
                    <span className="font-mono-code text-[10px] font-bold text-[#1E40AF] uppercase block">{m.label}</span>
                    <span className="font-mono-code text-base font-bold text-[#0F172A] mt-1 block">{m.val}</span>
                  </div>
                ))}
              </div>

              {benchmark.retrieval_comparison && (
                <div>
                  <h3 className="font-mono-code text-xs font-bold text-slate-700 uppercase mb-2">
                    1. So Sánh Truy Xuất Bằng Chứng: BM25 Baseline vs. Dense vs. Hybrid + Cross-Reranker
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-600 font-mono-code uppercase">
                          <th className="py-2.5 px-2">Phương pháp Truy xuất</th>
                          <th className="py-2.5 px-2">Recall@1</th>
                          <th className="py-2.5 px-2">Recall@3</th>
                          <th className="py-2.5 px-2">Recall@5</th>
                          <th className="py-2.5 px-2">MRR@5</th>
                          <th className="py-2.5 px-2">NDCG@5</th>
                          <th className="py-2.5 px-2">Ev.Precision</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono-code">
                        {Object.entries(benchmark.retrieval_comparison).map(([k, r]) => (
                          <tr key={k} className={k === 'hybrid_rerank' ? 'bg-blue-50/60 font-bold' : ''}>
                            <td className="py-2.5 px-2 text-[#1E40AF]">{r.method_name}</td>
                            <td className="py-2.5 px-2">{(r.recall_at_1 * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-2">{(r.recall_at_3 * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-2 text-emerald-700">{(r.recall_at_5 * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-2">{(r.mrr_at_5 * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-2">{(r.ndcg_at_5 * 100).toFixed(1)}%</td>
                            <td className="py-2.5 px-2">{(r.evidence_precision * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {benchmark.verification_comparison && (
                <div>
                  <h3 className="font-mono-code text-xs font-bold text-slate-700 uppercase mb-2">
                    2. So Sánh Suy Luận Kiểm Chứng 3 Nhãn (Supported / Refuted / NEI): NLI vs. LLM Few-Shot
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-600 font-mono-code uppercase">
                          <th className="py-2.5 px-2">Mô hình Suy luận (Verifier)</th>
                          <th className="py-2.5 px-2">Macro-F1</th>
                          <th className="py-2.5 px-2">Accuracy</th>
                          <th className="py-2.5 px-2">Precision</th>
                          <th className="py-2.5 px-2">Recall</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono-code">
                        {Object.entries(benchmark.verification_comparison).map(([k, v]) => (
                          <tr key={k} className={k === 'hybrid_consensus' ? 'bg-emerald-50/60 font-bold' : ''}>
                            <td className="py-2.5 px-2 text-[#0F172A]">{v.name}</td>
                            <td className="py-2.5 px-2 text-emerald-700">{(v.macro_f1 * 100).toFixed(2)}%</td>
                            <td className="py-2.5 px-2">{(v.accuracy * 100).toFixed(2)}%</td>
                            <td className="py-2.5 px-2">{(v.precision * 100).toFixed(2)}%</td>
                            <td className="py-2.5 px-2">{(v.recall * 100).toFixed(2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Pre-trained Benchmark Table */}
          {Object.keys(results).length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                Model Evaluation Registry
              </span>
              <h2 className="text-lg font-bold text-[#0F172A] mt-0.5 mb-4">
                Bảng Kết Quả Thực Nghiệm Các Mô Hình Đã Huấn Luyện
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-600 font-mono-code uppercase">
                      <th className="py-3 px-3">Mô hình</th>
                      <th className="py-3 px-3">Accuracy</th>
                      <th className="py-3 px-3">F1-Score</th>
                      <th className="py-3 px-3">Precision</th>
                      <th className="py-3 px-3">Recall</th>
                      <th className="py-3 px-3">FAR</th>
                      <th className="py-3 px-3">Thời gian</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono-code">
                    {Object.entries(results).map(([name, data]) => (
                      <tr key={name} className="hover:bg-slate-50 transition-colors">
                        <td className="py-3 px-3 font-bold text-[#1E40AF] uppercase">{name}</td>
                        <td className="py-3 px-3 font-bold text-[#0F172A]">{(data.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 font-semibold text-slate-800">{(data.f1 * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-slate-700">{(data.precision * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-emerald-700 font-semibold">{(data.recall * 100).toFixed(1)}%</td>
                        <td className="py-3 px-3 text-rose-700">{(data.false_alarm_rate * 100).toFixed(2)}%</td>
                        <td className="py-3 px-3 text-slate-500">{data.duration?.toFixed(1)}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingPage;
