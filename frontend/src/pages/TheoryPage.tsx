import { useState } from 'react';

import Chapter1NLP from './theory-chapters/Chapter1NLP';
import Chapter2Vector from './theory-chapters/Chapter2Vector';
import Chapter3Models from './theory-chapters/Chapter3Models';
import Chapter4Metrics from './theory-chapters/Chapter4Metrics';
import Chapter5Matrix from './theory-chapters/Chapter5Matrix';
import Chapter6Pipeline from './theory-chapters/Chapter6Pipeline';

const TheoryPage = () => {
  const [activeChapter, setActiveChapter] = useState('nlp');

  const chapters = [
    { id: 'nlp', code: '01', title: 'Tiền xử lý Tiếng Việt (NLP & Claim)' },
    { id: 'vector', code: '02', title: 'Biểu diễn Vector & Truy xuất (Mục S2.2)' },
    { id: 'models', code: '03', title: '8 Thuật toán Phân loại & Đối chiếu NLI' },
    { id: 'metrics', code: '04', title: 'Chỉ số Đánh giá & Độ mới lạ (Mục S7.2)' },
    { id: 'matrix', code: '05', title: 'Ma trận Nhầm lẫn (Confusion Matrix)' },
    { id: 'pipeline', code: '06', title: 'Kiến trúc Luồng 5 Giai đoạn (12 Node)' },
  ];

  const renderContent = () => {
    switch (activeChapter) {
      case 'nlp': return <Chapter1NLP />;
      case 'vector': return <Chapter2Vector />;
      case 'models': return <Chapter3Models />;
      case 'metrics': return <Chapter4Metrics />;
      case 'matrix': return <Chapter5Matrix />;
      case 'pipeline': return <Chapter6Pipeline />;
      default: return <Chapter1NLP />;
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 md:px-8 pb-20">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar Navigation */}
        <aside className="w-full lg:w-80 shrink-0">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 sticky top-24 space-y-2">
            <div className="pb-4 mb-3 border-b border-slate-100">
              <span className="font-mono-code text-xs font-bold text-[#1E40AF] uppercase">
                Documentation & Architecture
              </span>
              <h2 className="text-lg font-bold text-[#0F172A] mt-0.5">
                Cơ Sở Lý Thuyết Hệ Thống
              </h2>
            </div>

            {chapters.map((ch) => {
              const isSelected = activeChapter === ch.id;
              return (
                <button
                  key={ch.id}
                  type="button"
                  onClick={() => setActiveChapter(ch.id)}
                  className={`cursor-pointer w-full text-left px-4 py-3 rounded-xl text-xs font-semibold transition-colors duration-200 flex items-center gap-3 border focus-visible:ring-2 focus-visible:ring-[#1E40AF] outline-none ${
                    isSelected
                      ? 'bg-[#1E40AF] text-white border-[#1E40AF] shadow-sm'
                      : 'bg-[#F8FAFC] text-slate-700 border-slate-200/80 hover:border-[#3B82F6] hover:text-[#0F172A]'
                  }`}
                >
                  <span
                    className={`font-mono-code px-2 py-0.5 rounded text-[11px] font-bold ${
                      isSelected ? 'bg-white/20 text-white' : 'bg-slate-200 text-slate-700'
                    }`}
                  >
                    {ch.code}
                  </span>
                  <span className="leading-snug">{ch.title}</span>
                </button>
              );
            })}
          </div>
        </aside>

        {/* Main Content Panel */}
        <section className="flex-1 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-10 overflow-x-hidden">
          {renderContent()}
        </section>
      </div>
    </div>
  );
};

export default TheoryPage;
