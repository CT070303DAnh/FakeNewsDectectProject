

const Chapter5Matrix = () => {
  return (
    <div className="space-y-32 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <section>
        <h1 className="text-6xl font-black mb-6 text-slate-900 tracking-tight">5. Ma trận <span className="text-blue-600">Nhầm lẫn</span></h1>
        <p className="text-xl text-slate-500 leading-relaxed max-w-3xl font-medium">
          Dù tên là "Nhầm lẫn" (Confusion Matrix), đây lại là công cụ rành mạch nhất để vạch trần mọi cái "đúng", cái "sai", và cái "ngu ngơ" của một hệ thống AI.
        </p>
      </section>

      <div className="bg-slate-900 p-10 md:p-16 rounded-[4rem] shadow-2xl relative overflow-hidden">
         <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl -mt-20 -mr-20"></div>
         <div className="absolute bottom-0 left-0 w-64 h-64 bg-rose-500/10 rounded-full blur-3xl -mb-20 -ml-20"></div>

         <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 relative z-10 items-center">
            {/* Ma Trận UI */}
            <div className="flex flex-col items-center">
               <h3 className="text-[10px] uppercase font-black tracking-widest text-slate-400 mb-8">Bảng Đối Chiếu 2x2</h3>
               
               <div className="grid grid-cols-[auto_1fr_1fr] gap-4 w-full max-w-md">
                  {/* Góc trên trái (Trống) */}
                  <div></div>
                  <div className="text-center pb-2 border-b-2 border-slate-700/50">
                     <span className="text-emerald-400 font-black tracking-widest uppercase text-[10px] block">Thực tế</span>
                     <span className="text-white font-bold">REAL News</span>
                  </div>
                  <div className="text-center pb-2 border-b-2 border-slate-700/50">
                     <span className="text-rose-400 font-black tracking-widest uppercase text-[10px] block">Thực tế</span>
                     <span className="text-white font-bold">FAKE News</span>
                  </div>

                  {/* Hàng 1 */}
                  <div className="flex flex-col justify-center items-end pr-4 border-r-2 border-slate-700/50">
                     <span className="text-slate-400 font-black tracking-widest uppercase text-[10px] text-right">Mô hình Đoán</span>
                     <span className="text-white font-bold text-right pt-1">REAL</span>
                  </div>
                  <div className="bg-emerald-900/40 p-8 rounded-3xl border border-emerald-500/50 text-center relative group transition-all hover:scale-105 hover:bg-emerald-800/50 cursor-crosshair">
                     <div className="text-4xl font-black text-emerald-400">TN</div>
                     <div className="absolute -top-3 -right-3 w-8 h-8 bg-emerald-500 rounded-full text-white flex items-center justify-center font-bold text-xs shadow-lg shadow-emerald-500/50 border-2 border-slate-900">✓</div>
                  </div>
                  <div className="bg-rose-900/40 p-8 rounded-3xl border border-rose-500/50 text-center relative group transition-all hover:scale-105 hover:bg-rose-800/50 cursor-crosshair">
                     <div className="text-4xl font-black text-rose-400">FN</div>
                     <div className="absolute -top-3 -right-3 w-8 h-8 bg-rose-500 rounded-full text-white flex items-center justify-center font-bold text-[10px] shadow-lg shadow-rose-500/50 border-2 border-slate-900 uppercase">Sai</div>
                  </div>

                  {/* Hàng 2 */}
                  <div className="flex flex-col justify-center items-end pr-4 border-r-2 border-slate-700/50">
                     <span className="text-slate-400 font-black tracking-widest uppercase text-[10px] text-right">Mô hình Đoán</span>
                     <span className="text-white font-bold text-right pt-1">FAKE</span>
                  </div>
                  <div className="bg-rose-900/40 p-8 rounded-3xl border border-rose-500/50 text-center relative group transition-all hover:scale-105 hover:bg-rose-800/50 cursor-crosshair">
                     <div className="text-4xl font-black text-rose-400">FP</div>
                     <div className="absolute -top-3 -right-3 w-8 h-8 bg-rose-500 rounded-full text-white flex items-center justify-center font-bold text-[10px] shadow-lg shadow-rose-500/50 border-2 border-slate-900 uppercase">Sai</div>
                  </div>
                  <div className="bg-emerald-900/40 p-8 rounded-3xl border border-emerald-500/50 text-center relative group transition-all hover:scale-105 hover:bg-emerald-800/50 cursor-crosshair">
                     <div className="text-4xl font-black text-emerald-400">TP</div>
                     <div className="absolute -top-3 -right-3 w-8 h-8 bg-emerald-500 rounded-full text-white flex items-center justify-center font-bold text-xs shadow-lg shadow-emerald-500/50 border-2 border-slate-900">✓</div>
                  </div>
               </div>
            </div>

            {/* Giải nghĩa chi tiết */}
            <div className="space-y-6">
               <div className="p-6 bg-slate-800/50 rounded-3xl border border-slate-700/50 hover:bg-slate-800 transition-colors">
                  <div className="flex items-center gap-4 mb-2">
                     <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs font-black uppercase tracking-widest">TP (True Positive)</span>
                     <span className="text-white font-bold">Thành Công Cốt Lõi</span>
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed">
                     Bài đó <strong>thực sự là Tin Giả</strong>, và AI đã <strong>đoán đúng là Tin Giả</strong>. Đây là mục tiêu quan trọng nhất của hệ thống.
                  </p>
               </div>

               <div className="p-6 bg-slate-800/50 rounded-3xl border border-slate-700/50 hover:bg-slate-800 transition-colors">
                  <div className="flex items-center gap-4 mb-2">
                     <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded text-xs font-black uppercase tracking-widest">TN (True Negative)</span>
                     <span className="text-white font-bold">Bảo Vệ Người Vô Tội</span>
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed">
                     Bài đó là <strong>Tin Thật</strong>, và AI đã để yên, <strong>đoán đúng là Tin Thật</strong>. Người dùng bình thường không bị làm phiền.
                  </p>
               </div>

               <div className="p-6 bg-rose-900/20 rounded-3xl border border-rose-500/20 hover:bg-rose-900/40 transition-colors">
                  <div className="flex items-center gap-4 mb-2">
                     <span className="px-3 py-1 bg-rose-500/20 text-rose-400 rounded text-xs font-black uppercase tracking-widest">FP (False Positive)</span>
                     <span className="text-white font-bold">Án Oan (Báo Động Giả)</span>
                  </div>
                  <p className="text-sm text-rose-200/70 leading-relaxed">
                     Bài đó là <strong>Tin Thật</strong>, nhưng AI "ngáo" lại <strong>đoán nhầm thành Tin Giả</strong>. Chuyện này gây ức chế cho người dùng vì bị chặn vô cớ.
                  </p>
               </div>

               <div className="p-6 bg-orange-900/20 rounded-3xl border border-orange-500/20 hover:bg-orange-900/40 transition-colors">
                  <div className="flex items-center gap-4 mb-2">
                     <span className="px-3 py-1 bg-orange-500/20 text-orange-400 rounded text-xs font-black uppercase tracking-widest">FN (False Negative)</span>
                     <span className="text-white font-bold">Kẻ Thù Lọt Lưới</span>
                  </div>
                  <p className="text-sm text-orange-200/70 leading-relaxed">
                     Bài đó là <strong>Tin Giả</strong>, nhưng AI bị lừa và <strong>đoán nhầm thành Tin Thật</strong>. Tin giả tiếp tục phát tán, cực kỳ nguy hiểm cho cộng đồng.
                  </p>
               </div>
            </div>
         </div>
      </div>

      <div className="bg-white p-10 md:p-14 rounded-[4rem] shadow-sm border border-slate-100 flex flex-col md:flex-row gap-12 items-center">
         <div className="flex-1 space-y-6">
            <h3 className="text-3xl font-black text-slate-900">Tại sao lại cần cái bảng này?</h3>
            <p className="text-slate-600 leading-relaxed text-lg">
               Vì Accuracy (Độ chính xác) bị mù loà trước sự chênh lệch dữ liệu.
            </p>
            <div className="space-y-4 text-sm text-slate-600 font-medium">
               <p className="flex gap-3 items-start">
                  <span className="text-rose-500 font-bold mt-0.5">⚠️</span>
                  <span>Nếu bạn có 1000 bài viết (990 Thật, 10 Giả). Mô hình của bạn bị hỏng và cứ phán bừa: <em className="text-slate-900 font-bold">"Tất cả đều là tin Thật!"</em></span>
               </p>
               <p className="flex gap-3 items-start">
                  <span className="text-rose-500 font-bold mt-0.5">📈</span>
                  <span>Accuracy lúc này đạt <strong>990/1000 = 99%</strong> (Một con số mơ ước!).</span>
               </p>
               <p className="flex gap-3 items-start">
                  <span className="text-emerald-500 font-bold mt-0.5">💡</span>
                  <span>Nhưng nếu nhìn vào Ma trận: <strong>TP = 0</strong> (Chả bắt được cái tin giả nào), và <strong>FN = 10</strong> (Bỏ lọt sạch sành sanh). Nhờ Confusion Matrix, bạn lập tức nhận ra Mô hình này thực chất là <strong>đồ bỏ đi rác rưởi</strong>.</span>
               </p>
            </div>
         </div>
         <div className="w-full md:w-1/3 shrink-0 bg-rose-50 p-8 rounded-3xl border border-rose-100 text-center">
            <div className="text-5xl mb-4">🕵️‍♂️</div>
            <h5 className="font-black text-rose-900 mb-2">Lời Khuyên AI</h5>
            <p className="text-xs font-bold text-rose-700/80 leading-relaxed">
               Luôn luôn in Ma Trận Nhầm Lẫn ra sau khi Evaluate. Mọi mánh khóe về dữ liệu lệch (Imbalanced Data) đều bị vạch trần ở đây.
            </p>
         </div>
      </div>
    </div>
  );
};

export default Chapter5Matrix;
