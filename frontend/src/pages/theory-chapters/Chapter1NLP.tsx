

const Chapter1NLP = () => {
  return (
    <div className="space-y-32 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <section>
        <h1 className="text-6xl font-black mb-6 text-slate-900 tracking-tight">1. Tiền xử lý <span className="text-blue-600">Ngôn ngữ Tự nhiên</span></h1>
        <p className="text-xl text-slate-500 leading-relaxed max-w-3xl font-medium">
          Dữ liệu văn bản thô giống như một quặng đá chứa cả vàng lẫn đất cát. Ngôn ngữ tự nhiên (NLP) cung cấp các công cụ để "tinh chế" văn bản, giúp mô hình AI chỉ tập trung vào những ý nghĩa cốt lõi.
        </p>
      </section>

      {/* --- 1. TẠI SAO PHẢI TIỀN XỬ LÝ --- */}
      <div className="space-y-12">
         <div className="flex items-center gap-6">
           <div className="w-16 h-16 bg-sky-500 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-sky-200">1</div>
           <div>
             <h2 className="text-4xl font-black text-slate-900">Mục đích (Concept)</h2>
             <p className="text-sky-500 font-bold uppercase text-xs tracking-widest mt-1">Vì sao AI không hiểu chữ?</p>
           </div>
         </div>
         
         <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
            <div className="space-y-8">
               <p className="text-slate-600 leading-relaxed">
                  Máy tính chỉ có thể xử lý các con số (0 và 1). Nó không hiểu "tốt" khác "xấu" như thế nào cho đến khi chúng ta biến những từ đó thành một dạng biểu diễn toán học. Tuy nhiên, trước khi biến thành số, chúng ta phải "dọn dẹp" để giảm kích thước từ vựng và loại bỏ nhiễu.
               </p>
               <div className="grid grid-cols-2 gap-4">
                  <div className="p-5 bg-white rounded-3xl border border-slate-100 shadow-sm">
                     <h5 className="font-black text-rose-500 text-xs uppercase mb-2">Dữ liệu thô (Nhiễu)</h5>
                     <p className="text-xs text-slate-500 font-medium">Dấu câu, Icon, HTML tags, từ nối (và, thì, là, mà), viết hoa chữ thường lộn xộn.</p>
                  </div>
                  <div className="p-5 bg-white rounded-3xl border border-slate-100 shadow-sm">
                     <h5 className="font-black text-teal-500 text-xs uppercase mb-2">Dữ liệu sạch (Tinh khiết)</h5>
                     <p className="text-xs text-slate-500 font-medium">Chỉ còn lại danh từ, động từ, tính từ mang ý nghĩa quan trọng nhất.</p>
                  </div>
               </div>
            </div>

            <div className="bg-slate-900 p-8 rounded-[3rem] text-white shadow-2xl relative">
               <div className="absolute top-0 right-0 w-32 h-32 bg-sky-500/10 rounded-full blur-3xl"></div>
               <h3 className="text-[10px] font-black tracking-widest text-slate-500 uppercase mb-6">Mô phỏng: Rác → Vàng</h3>
               <div className="bg-slate-800 p-4 rounded-2xl border border-slate-700 font-mono text-sm break-all text-slate-400 mb-6">
                  <span className="text-rose-400">!!!</span> HOT <span className="text-rose-400">&lt;br&gt;</span> CHẤN ĐỘNG <span className="text-rose-400">@vietnam</span>: Thấy alien ở 
                  công viên hôm qua <span className="text-rose-400">!!! =)))</span>
               </div>
               <div className="flex justify-center mb-6">
                  <div className="w-8 h-8 rounded-full bg-sky-500 flex items-center justify-center animate-bounce shadow-[0_0_15px_#0ea5e9]">
                     <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>
                  </div>
               </div>
               <div className="bg-emerald-900/30 p-4 rounded-2xl border border-emerald-500/30 font-mono text-sm break-all text-emerald-400">
                  hot chấn động vietnam thấy alien công viên hôm qua
               </div>
            </div>
         </div>
      </div>

      {/* --- 2. WORD SEGMENTATION --- */}
      <div className="space-y-12">
         <div className="flex items-center gap-6">
           <div className="w-16 h-16 bg-indigo-500 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-indigo-200">2</div>
           <div>
             <h2 className="text-4xl font-black text-slate-900">Tách từ (Word Segmentation)</h2>
             <p className="text-indigo-500 font-bold uppercase text-xs tracking-widest mt-1">Đặc thù của Tiếng Việt (Underthesea)</p>
           </div>
         </div>

         <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
            <div className="bg-indigo-50/50 p-8 rounded-[3rem] border border-indigo-100 flex flex-col order-2 lg:order-1">
               <h4 className="text-lg font-black text-slate-900 mb-6 flex items-center gap-2">
                 <span className="w-2 h-6 bg-indigo-500 rounded-full"></span> ⚙️ Nguyên lý hoạt động
               </h4>
               <p className="text-sm text-slate-600 font-medium mb-6">Khác với Tiếng Anh (khoảng trắng = 1 từ), Tiếng Việt có các từ ghép 2-3 âm tiết. Thư viện <strong>underthesea</strong> sử dụng học máy (CRF hoặc BiLSTM) để gỡ rối các giới hạn từ.</p>
               
               <div className="space-y-4">
                  <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden group">
                     <div className="absolute top-0 left-0 w-1 h-full bg-rose-400 group-hover:w-2 transition-all"></div>
                     <span className="text-[10px] uppercase font-black tracking-widest text-slate-400">Tách bằng khoảng trắng (Sai)</span>
                     <div className="mt-2 text-sm font-mono flex gap-2 flex-wrap">
                        <span className="px-2 py-1 bg-slate-100 rounded">cử</span>
                        <span className="px-2 py-1 bg-slate-100 rounded">tri</span>
                        <span className="px-2 py-1 bg-slate-100 rounded">bầu</span>
                        <span className="px-2 py-1 bg-slate-100 rounded">cử</span>
                     </div>
                  </div>
                  <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden group">
                     <div className="absolute top-0 left-0 w-1 h-full bg-emerald-400 group-hover:w-2 transition-all"></div>
                     <span className="text-[10px] uppercase font-black tracking-widest text-slate-400">Tách bằng Underthesea (Đúng)</span>
                     <div className="mt-2 text-sm font-mono flex gap-2 flex-wrap">
                        <span className="px-2 py-1 bg-emerald-50 text-emerald-700 rounded border border-emerald-100">cử_tri</span>
                        <span className="px-2 py-1 bg-emerald-50 text-emerald-700 rounded border border-emerald-100">bầu_cử</span>
                     </div>
                  </div>
               </div>
            </div>

            <div className="space-y-6 order-1 lg:order-2">
               <section className="space-y-4">
                 <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                   <span className="w-2 h-6 bg-indigo-500 rounded-full"></span> 🟢 Vấn đề "Boundary Ambiguity"
                 </h4>
                 <p className="text-slate-600 leading-relaxed">
                   Nếu không tách đúng từ, một câu có thể bị hiểu lầm hoàn toàn. Việc nối các âm tiết của một từ bằng dấu gạch dưới `_` giúp AI tính toán chính xác tần suất một NGUYÊN TỪ xuất hiện.
                 </p>
               </section>
               <div className="mb-4">
                 <h4 className="font-bold text-slate-800 text-sm mb-2">Công thức phân giải từ (CRF Score) mô phỏng:</h4>
                 <div className="bg-slate-900 p-4 rounded-2xl text-slate-300 font-serif italic text-sm overflow-x-auto">
                    P(y | x) = (1 / Z(x)) * exp( Σ [ λ_i * f_i(y_t, y_&#123;t-1&#125;, x, t) ] )
                 </div>
                 <div className="text-[10px] text-slate-500 mt-2">
                    * Trong đó <strong>f_i</strong> là các hàm đặc trưng (VD: 2 âm tiết này có hay đứng cùng nhau trong từ điển không).
                 </div>
               </div>
            </div>
         </div>
      </div>

      {/* --- 3. STOP WORDS VÀ LÀM SẠCH BIỂU THỨC --- */}
      <div className="space-y-12">
         <div className="flex items-center gap-6">
           <div className="w-16 h-16 bg-emerald-500 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-emerald-200">3</div>
           <div>
             <h2 className="text-4xl font-black text-slate-900">Stop-words & Regex</h2>
             <p className="text-emerald-500 font-bold uppercase text-xs tracking-widest mt-1">Ép cân cho bộ dữ liệu</p>
           </div>
         </div>

         <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white p-8 rounded-[3rem] shadow-sm border border-slate-100 space-y-4">
               <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                 <span className="w-2 h-6 bg-emerald-500 rounded-full"></span> 🛑 Từ dừng (Stopwords)
               </h4>
               <p className="text-sm text-slate-600 leading-relaxed">
                 Trong mọi ngôn ngữ, có những từ xuất hiện với tần suất khổng lồ nhưng không mang lại ý nghĩa phân loại. Ví dụ: "rằng", "thì", "là", "mà", "của". Xóa chúng giúp giảm kích thước Ma trận Vectơ lên đến 40%.
               </p>
               <div className="bg-slate-50 p-4 rounded-2xl font-mono text-xs flex flex-wrap gap-2">
                  <span className="line-through text-slate-400">thì</span>
                  <span className="line-through text-slate-400">những</span>
                  <span className="line-through text-slate-400">được</span>
                  <span className="px-2 py-1 bg-emerald-100 text-emerald-700 font-bold rounded">tổng_thống</span>
                  <span className="line-through text-slate-400">tại</span>
               </div>
            </div>

            <div className="bg-white p-8 rounded-[3rem] shadow-sm border border-slate-100 space-y-4">
               <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                 <span className="w-2 h-6 bg-slate-800 rounded-full"></span> ✂️ Regular Expression (Regex)
               </h4>
               <p className="text-sm text-slate-600 leading-relaxed">
                  Regex là các mẫu toán học dùng để "truy bắt" các chuỗi văn bản không theo quy tắc chuẩn. Tin giả thường lạm dụng các ký hiệu đặc biệt để giật gân, cần phải được khử trùng.
               </p>
               <div className="bg-slate-900 p-4 rounded-2xl font-mono text-xs flex flex-col gap-2">
                  <div className="flex justify-between items-center bg-slate-800 p-2 rounded">
                     <span className="text-rose-400">http\S+</span>
                     <span className="text-slate-400 text-[10px] uppercase font-bold">Xóa Link</span>
                  </div>
                  <div className="flex justify-between items-center bg-slate-800 p-2 rounded">
                     <span className="text-amber-400">[^\w\s]</span>
                     <span className="text-slate-400 text-[10px] uppercase font-bold">Xóa Emoji/Ký hiệu</span>
                  </div>
               </div>
            </div>
         </div>
      </div>
    </div>
  );
};

export default Chapter1NLP;
