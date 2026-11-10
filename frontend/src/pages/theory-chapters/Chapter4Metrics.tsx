

const Chapter4Metrics = () => {
  return (
    <div className="space-y-32 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <section>
        <h1 className="text-6xl font-black mb-6 text-slate-900 tracking-tight">4. Thước đo <span className="text-blue-600">Hiệu năng</span></h1>
        <p className="text-xl text-slate-500 leading-relaxed max-w-3xl font-medium">
          "Đúng 99%" đôi khi vẫn là một con số lừa dối. Trong bài toán phân loại thực tế, chúng ta cần nhiều góc nhìn khác nhau để biết mô hình AI có thực sự đáng tin cậy hay không.
        </p>
      </section>

      {/* --- CƠ SỞ CHÍNH --- */}
      <div className="bg-slate-900 p-8 md:p-12 rounded-[3.5rem] shadow-2xl space-y-8 relative overflow-hidden">
         <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl -mt-20 -mr-20"></div>
         <h4 className="text-lg font-black text-slate-300 flex items-center gap-2 mb-8 relative z-10">
           <span className="w-2 h-6 bg-blue-500 rounded-full"></span> ⚙️ Bốn cột trụ cơ sở (T - F - P - N)
         </h4>
         <div className="grid grid-cols-2 md:grid-cols-4 gap-6 relative z-10">
            <div className="bg-slate-800 p-6 rounded-3xl border border-slate-700 text-center">
               <div className="text-3xl font-black text-green-400 mb-2">TP</div>
               <p className="text-[10px] uppercase font-bold text-slate-500">True Positive<br/>Đoán Fake & Thật sự Fake</p>
            </div>
            <div className="bg-slate-800 p-6 rounded-3xl border border-slate-700 text-center">
               <div className="text-3xl font-black text-slate-300 mb-2">TN</div>
               <p className="text-[10px] uppercase font-bold text-slate-500">True Negative<br/>Đoán Thật & Thật sự Thật</p>
            </div>
            <div className="bg-slate-800 p-6 rounded-3xl border border-slate-700 text-center">
               <div className="text-3xl font-black text-rose-500 mb-2">FP</div>
               <p className="text-[10px] uppercase font-bold text-slate-500">False Positive<br/>Đoán Fake NHƯNG Thật sự Thật</p>
            </div>
            <div className="bg-slate-800 p-6 rounded-3xl border border-slate-700 text-center">
               <div className="text-3xl font-black text-yellow-500 mb-2">FN</div>
               <p className="text-[10px] uppercase font-bold text-slate-500">False Negative<br/>Đoán Thật NHƯNG Thật sự Fake</p>
            </div>
         </div>
      </div>

      {/* --- CÁC CHỈ SỐ --- */}
      <div className="space-y-16">
         {/* 1. ACCURACY */}
         <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div className="bg-sky-50 p-10 rounded-[3rem] border border-sky-100 flex flex-col justify-center">
               <div className="flex items-center gap-4 mb-6">
                 <div className="w-12 h-12 rounded-[1.5rem] bg-sky-500 text-white flex items-center justify-center font-black text-2xl">1</div>
                 <h3 className="text-3xl font-black text-slate-900">Accuracy</h3>
                 <span className="px-3 py-1 bg-white text-sky-600 rounded-full text-[10px] font-black uppercase shadow-sm">Độ chính xác</span>
               </div>
               <div className="bg-white p-6 rounded-3xl font-mono text-sm text-center border border-sky-200 shadow-sm mb-6 text-slate-600">
                  <span className="text-green-600 font-bold">TP</span> + <span className="text-green-600 font-bold">TN</span> <br/><span className="border-t border-slate-300 px-4 mt-1 inline-block">Tổng số dự đoán</span>
               </div>
               <p className="text-sm text-slate-600 font-medium">Là tỷ lệ những lần AI nói ĐÚNG (bất kể Fake hay Real) trên toàn bộ bài test.</p>
            </div>
            <div className="p-8">
               <h5 className="text-[10px] uppercase tracking-widest text-rose-500 font-black mb-4">Lỗ hổng (Paradox)</h5>
               <p className="text-slate-600 text-sm leading-relaxed mb-4">
                  Trong dữ liệu thực tế, Tin rác/Fake chỉ chiếm 5%, Tin thật chiếm 95%. Nếu AI bị lỗi và kết luận <strong>"Tất cả đều là Tin thật"</strong>, Accuracy của nó vẫn đạt <strong>95%</strong> dù nó chả phát hiện được bài Fake nào. Do đó, chúng ta cần Precision và Recall.
               </p>
            </div>
         </div>

         {/* 2. PRECISION */}
         <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div className="p-8 order-2 md:order-1">
               <h5 className="text-[10px] uppercase tracking-widest text-indigo-500 font-black mb-4">Ý NGHĨA THỰC TẾ</h5>
               <p className="text-slate-600 text-sm leading-relaxed font-bold italic mb-2">"Nhầm bạn hơn bỏ sót kẻ thù?" Không, Precision là ngược lại.</p>
               <p className="text-slate-600 text-sm leading-relaxed">
                  Trọng tâm là <strong>Sự thận trọng</strong>. Nếu Precision cao, tức là AI rất cẩn thận, mỗi khi nó "mở miệng" nói tin này là giả, thì chắc chắn 99% đó là giả. Nếu Precision thấp, AI sẽ khoá nhầm tài khoản của những người đăng tin bình thường (Báo động giả FP).
               </p>
            </div>
            <div className="bg-indigo-50 p-10 rounded-[3rem] border border-indigo-100 flex flex-col justify-center order-1 md:order-2">
               <div className="flex items-center gap-4 mb-6">
                 <div className="w-12 h-12 rounded-[1.5rem] bg-indigo-500 text-white flex items-center justify-center font-black text-2xl">2</div>
                 <h3 className="text-3xl font-black text-slate-900">Precision</h3>
                 <span className="px-3 py-1 bg-white text-indigo-600 rounded-full text-[10px] font-black uppercase shadow-sm">Độ Chuẩn Xác</span>
               </div>
               <div className="bg-white p-6 rounded-3xl font-mono text-sm text-center border border-indigo-200 shadow-sm mb-6 text-slate-600">
                  <span className="text-green-600 font-bold">TP</span><br/><span className="border-t border-slate-300 px-4 mt-1 inline-block"><span className="text-green-600 font-bold">TP</span> + <span className="text-rose-500 font-bold">FP</span></span>
               </div>
               <p className="text-sm text-slate-600 font-medium">Trong mớ bài mà AI bảo là "Tin giả", có bao nhiêu % đúng là thật sự Giả?</p>
            </div>
         </div>

         {/* 3. RECALL */}
         <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            <div className="bg-emerald-50 p-10 rounded-[3rem] border border-emerald-100 flex flex-col justify-center">
               <div className="flex items-center gap-4 mb-6">
                 <div className="w-12 h-12 rounded-[1.5rem] bg-emerald-500 text-white flex items-center justify-center font-black text-2xl">3</div>
                 <h3 className="text-3xl font-black text-slate-900">Recall</h3>
                 <span className="px-3 py-1 bg-white text-emerald-600 rounded-full text-[10px] font-black uppercase shadow-sm">Độ Bao Phủ</span>
               </div>
               <div className="bg-white p-6 rounded-3xl font-mono text-sm text-center border border-emerald-200 shadow-sm mb-6 text-slate-600">
                  <span className="text-green-600 font-bold">TP</span><br/><span className="border-t border-slate-300 px-4 mt-1 inline-block"><span className="text-green-600 font-bold">TP</span> + <span className="text-yellow-600 font-bold">FN</span></span>
               </div>
               <p className="text-sm text-slate-600 font-medium">Trong tổng số Tin giả đang trôi nổi trên mạng, AI đã bắt mạch được bao nhiêu %?</p>
            </div>
            <div className="p-8">
               <h5 className="text-[10px] uppercase tracking-widest text-emerald-600 font-black mb-4">Ý NGHĨA THỰC TẾ</h5>
               <p className="text-slate-600 text-sm leading-relaxed font-bold italic mb-2">"Nhầm bạn còn hơn bỏ sót kẻ thù!"</p>
               <p className="text-slate-600 text-sm leading-relaxed mb-4">
                  Trọng tâm là <strong>Độ truy quét</strong>. Nếu Recall thấp (bỏ sót nhiều - FN), tin giả vẫn lan truyền trên mạng và gây hại. Thông thường, AI sẽ phải hy sinh một chút Precision để đẩy Recall lên cao (thà cảnh báo nhầm còn hơn để tin giả lọt lưới).
               </p>
            </div>
         </div>

         {/* 4. F1-SCORE */}
         <div className="bg-gradient-to-br from-violet-600 to-fuchsia-600 p-12 rounded-[4rem] text-white shadow-2xl relative overflow-hidden">
             <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-20 -mt-20 blur-3xl"></div>
             
             <div className="grid grid-cols-1 md:grid-cols-2 gap-12 relative z-10">
                <div>
                   <div className="flex items-center gap-4 mb-8">
                     <div className="w-12 h-12 rounded-[1.5rem] bg-white text-fuchsia-600 flex items-center justify-center font-black text-2xl shadow-xl">4</div>
                     <h3 className="text-4xl font-black">F1-Score</h3>
                   </div>
                   <p className="text-fuchsia-100 leading-relaxed text-lg font-medium mb-6">
                      Sự cân bằng Hoàn hảo. Trong thực tế, Precision và Recall luôn trực diện đối đầu nhau. Tăng cái này sẽ làm giảm cái kia.
                   </p>
                   <p className="text-white/80 leading-relaxed text-sm">
                      F1-Score là <strong>Trung bình điều hòa</strong> (Harmonic Mean) của cả 2 thằng. Khi đánh giá thuật toán nào tốt nhất trong bài toán phát hiện Fake News, F1-Score là vị vua đích thực.
                   </p>
                </div>
                <div className="flex flex-col items-center justify-center gap-6">
                   <div className="bg-white/10 p-8 rounded-[2rem] border border-white/20 font-mono text-xl text-center backdrop-blur-md">
                      2 × <span className="border-t border-white/30 px-4 mt-2 inline-block relative top-3"><span className="absolute -top-7 left-1/2 -translate-x-1/2 w-max">Precision × Recall</span>Precision + Recall</span>
                   </div>
                   <div className="flex items-center gap-4 text-xs font-black uppercase tracking-widest text-fuchsia-300">
                      <span>Cao nhất = 1.0</span>
                      <span className="w-1 h-1 bg-fuchsia-300 rounded-full"></span>
                      <span>Thấp nhất = 0.0</span>
                   </div>
                </div>
             </div>
         </div>

      </div>
    </div>
  );
};

export default Chapter4Metrics;
