

const Chapter2Vector = () => {
  return (
    <div className="space-y-32 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <section>
        <h1 className="text-6xl font-black mb-6 text-slate-900 tracking-tight">2. Vectơ hóa: Biến chữ thành <span className="text-blue-600">Toán học</span></h1>
        <p className="text-xl text-slate-500 leading-relaxed max-w-3xl font-medium">
          Dù đã được làm sạch, AI vẫn "mù" chữ. Để AI có thể tính toán, phân loại hay so sánh cường độ của cảm xúc, văn bản cần phải được biểu diễn dưới dạng các con số trong Không gian Vectơ N-chiều. Dưới đây là 2 phương pháp mạnh nhất.
        </p>
      </section>

      {/* --- 1. TF-IDF --- */}
      <div className="space-y-12">
        <div className="flex items-center justify-between">
           <div className="flex items-center gap-6">
             <div className="w-16 h-16 bg-blue-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-blue-200">1</div>
             <div>
               <h2 className="text-4xl font-black text-slate-900">Mô hình TF-IDF</h2>
               <p className="text-blue-600 font-bold uppercase text-xs tracking-widest mt-1">Cách mạng Thống kê học văn bản</p>
             </div>
           </div>
           <span className="hidden md:inline-block px-4 py-1 bg-blue-100 text-blue-700/80 rounded-full text-[10px] font-black uppercase">Term Frequency - Inverse Document Frequency</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="space-y-8">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-blue-600 rounded-full"></span> 🟢 Nguyên lý "Tần suất & Sự hiếm lạ"
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Đếm số lần xuất hiện của một từ (như mô hình Bag-of-Words cũ) là không đủ. Một từ tốt phải thỏa mãn 2 điều kiện: <strong>Xuất hiện nhiều trong bài viết hiện tại (TF) nhưng Phải hiếm gặp trong các bài viết khác (IDF).</strong>
              </p>
            </section>
            
            <section className="space-y-4">
               <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                 <span className="w-2 h-6 bg-blue-600 rounded-full"></span> ⚙️ Ví dụ thực tế
               </h4>
               <p className="text-sm text-slate-600 font-medium">
                  Giả sử có 100 bài báo. Chữ "tôi" xuất hiện trong 90 bài =&gt; Điểm IDF rất thấp (vô giá trị). Ngược lại, chữ "vaccine_giả" chỉ xuất hiện ở 3 bài, nhưng trong 1 bài nó lặp lại tới 5 lần =&gt; Điểm TF-IDF sẽ cực kỳ cao.
               </p>
             </section>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Tính toán cực nhanh, siêu hiệu quả trên văn bản chuyên ngành/tin tức có từ khóa đặc thù.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Không hiểu ngữ cảnh, đồng nghĩa, trái nghĩa hay thứ tự từ.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-8 rounded-[3rem] text-white shadow-2xl relative overflow-hidden flex flex-col justify-center">
             <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-blue-500/20 rounded-full blur-3xl"></div>
             <h4 className="text-xs font-black text-blue-400 uppercase mb-10 tracking-widest text-center">Bóc tách Công thức toán học</h4>
             <div className="space-y-8 relative z-10 w-full">
                <div className="flex flex-col md:flex-row items-center justify-around gap-8">
                  <div className="text-center w-full">
                    <div className="text-xl font-serif mb-4 flex justify-center">
                       <span className="text-white">TF(t, d)</span> <span className="text-blue-400 mx-2">=</span> <div className="flex flex-col"><span className="border-b border-white/30 pb-1">n<sub className="text-[10px]">t,d</sub></span><span className="pt-1">Σ n<sub className="text-[10px]">k,d</sub></span></div>
                    </div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase">Tần suất từ trong bài t</p>
                  </div>
                  <div className="text-3xl font-black text-blue-500/50">×</div>
                  <div className="text-center w-full">
                    <div className="text-xl font-serif mb-4 flex justify-center items-center">
                       <span className="text-white">IDF(t)</span> <span className="text-blue-400 mx-2">=</span> <span>log(</span><div className="flex flex-col mx-2"><span className="border-b border-white/30 pb-1">N</span><span className="pt-1">df<sub className="text-[10px]">t</sub></span></div><span>)</span>
                    </div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase">Độ bù hiếm toàn kho</p>
                  </div>
                </div>
                
                <div className="pt-8 border-t border-slate-700/50">
                    <div className="bg-slate-800 p-4 rounded-2xl flex flex-col items-center gap-3">
                       <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Ma trận Đầu ra</span>
                       <div className="font-mono text-xs flex gap-4 text-emerald-400">
                          <span>["vaccine", 5.2]</span>
                          <span>["tin_vịt", 4.8]</span>
                          <span className="text-slate-500">["là", 0.0]</span>
                       </div>
                    </div>
                </div>
             </div>
          </div>
        </div>
      </div>

      <div className="w-full h-px bg-slate-200 my-16"></div>

      {/* --- 2. DOC2VEC --- */}
      <div className="space-y-12">
        <div className="flex items-center justify-between">
           <div className="flex items-center gap-6">
             <div className="w-16 h-16 bg-indigo-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-indigo-200">2</div>
             <div>
               <h2 className="text-4xl font-black text-slate-900">Mô hình Doc2Vec</h2>
               <p className="text-indigo-600 font-bold uppercase text-xs tracking-widest mt-1">Deep Learning & Ngữ nghĩa</p>
             </div>
           </div>
           <span className="hidden md:inline-block px-4 py-1 bg-indigo-100 text-indigo-700/80 rounded-full text-[10px] font-black uppercase">Document to Vector</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="bg-indigo-900 p-10 rounded-[3rem] text-white shadow-2xl relative overflow-hidden order-2 lg:order-1">
             <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -mr-20 -mt-20 blur-3xl"></div>
             <h5 className="text-[10px] font-black text-indigo-300 uppercase tracking-widest mb-10 text-center">Kiến trúc Mạng Nơ-ron (PV-DM)</h5>
             
             <div className="flex justify-between items-center px-4 relative">
                {/* Lớp Input */}
                <div className="flex flex-col gap-3 z-10">
                   <div className="bg-indigo-700/50 p-3 rounded-xl border border-indigo-500/30 text-center">
                      <div className="text-[10px] font-black tracking-widest mb-1">DOC ID</div>
                      <div className="w-4 h-4 bg-yellow-400 rounded-full mx-auto shadow-[0_0_10px_#facc15]"></div>
                   </div>
                   <div className="bg-indigo-700/50 p-3 rounded-xl border border-indigo-500/30 text-[10px]">Từ 1</div>
                   <div className="bg-indigo-700/50 p-3 rounded-xl border border-indigo-500/30 text-[10px]">Từ 2</div>
                   <div className="bg-indigo-700/50 p-3 rounded-xl border border-indigo-500/30 text-[10px]">Từ 3</div>
                </div>

                {/* Các đường Line nối */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
                    <line x1="20%" y1="20%" x2="50%" y2="50%" stroke="white" strokeWidth="2" />
                    <line x1="20%" y1="40%" x2="50%" y2="50%" stroke="white" strokeWidth="2" />
                    <line x1="20%" y1="60%" x2="50%" y2="50%" stroke="white" strokeWidth="2" />
                    <line x1="20%" y1="80%" x2="50%" y2="50%" stroke="white" strokeWidth="2" />
                    
                    <line x1="50%" y1="50%" x2="80%" y2="50%" stroke="white" strokeWidth="2" />
                </svg>

                {/* Hidden / Projection Layer */}
                <div className="z-10 bg-indigo-500/50 h-32 w-12 rounded-full border border-indigo-400/50 flex items-center justify-center text-center p-2 text-[10px] font-bold leading-tight">
                   Lớp<br/>Ẩn
                </div>

                {/* Output Layer */}
                <div className="z-10 flex flex-col gap-2">
                   <div className="bg-indigo-700/50 px-4 py-8 rounded-xl border border-indigo-500/30 text-center relative overflow-hidden">
                      <div className="absolute inset-0 bg-green-500/20"></div>
                      <div className="text-[10px] font-black text-green-300 relative z-10">Từ<br/>Mục tiêu</div>
                   </div>
                </div>
             </div>
             
             <div className="mt-12 bg-indigo-950 p-4 rounded-2xl border border-indigo-800 text-center">
                <p className="text-xs text-indigo-300 font-medium leading-relaxed">
                   Khi mô hình học xong cách dự đoán "từ tiếp theo", cái cục <strong className="text-yellow-400">DOC ID</strong> (Id của bài báo) sẽ tự động lưu giữ <strong className="text-white">Toàn bộ Ngữ nghĩa</strong> của bài báo đó ở dạng Vectơ.
                </p>
             </div>
          </div>

          <div className="space-y-8 order-1 lg:order-2">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-indigo-600 rounded-full"></span> 🟢 Vượt qua giới hạn của đếm từ
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Máy tính không biết "Giết", "Sát hại", "Thủ tiêu" là giống nhau. Nhưng Doc2Vec học qua một Mạng Nơ-ron nhân tạo, khi thấy các từ này hay xuất hiện trong cùng một bối cảnh hình sự, nó sẽ xếp bộ số của chúng gần nhau trong Không gian Vectơ.
              </p>
            </section>

            <section className="space-y-4">
               <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                 <span className="w-2 h-6 bg-indigo-600 rounded-full"></span> 📐 PV-DM (Distributed Memory)
               </h4>
               <p className="text-sm text-slate-600 font-medium leading-relaxed">
                 Trong kiến trúc PV-DM (Paragraph Vector), mô hình giống như đang chơi trò "Điền từ vào chỗ trống". Nó nhìn vào các từ xung quanh và ID của bài viết để đoán từ bị thiếu. Nhờ ép bản thân phải đoán đúng, ID bài viết dần mang "linh hồn", hay phong cách hành văn của báo đó.
               </p>
             </section>

             <div className="grid grid-cols-2 gap-4">
               <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                 <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                 <p className="text-xs font-bold text-slate-700 leading-relaxed">Hiểu sâu xa về ngữ cảnh, phong cách viết. Nhận dạng được tin giả bị "xào nấu" ngôn từ.</p>
               </div>
               <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                 <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                 <p className="text-xs font-bold text-slate-700 leading-relaxed">Tốn kém tính toán, phải train mô hình rất nặng trên lượng data lớn.</p>
               </div>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chapter2Vector;
