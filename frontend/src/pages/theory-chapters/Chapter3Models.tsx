import { useState } from 'react';

const Chapter3Models = () => {
  // --- 1. HỒI QUY LOGISTIC STATES ---
  const [lrW, setLrW] = useState(1.5);
  const [lrB, setLrB] = useState(0);
  const [lrX, setLrX] = useState(1);
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
  const lrZ = lrW * lrX + lrB;
  const lrP = sigmoid(lrZ);

  // --- 2. SVM STATES ---
  const [svmW, setSvmW] = useState(1.5); 
  const marginSize = 50 / svmW; 

  // --- 3. CÂY QUYẾT ĐỊNH STATES ---
  const [dtSplit, setDtSplit] = useState(50);
  const calculateEntropy = (p: number) => {
    if (p <= 0 || p >= 1) return 0;
    return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
  };
  const totalPoints = 10;
  const leftCount = Math.max(1, Math.floor((dtSplit / 100) * totalPoints));
  const rightCount = totalPoints - leftCount;
  const leftRed = Math.min(leftCount, 2); 
  const rightRed = 2 - leftRed;
  const entropyLeft = calculateEntropy(leftRed / leftCount);

  // --- 4. NAIVE BAYES STATES ---
  const [nbWord1, setNbWord1] = useState(0.7); 
  const [nbWord2, setNbWord2] = useState(0.6); 
  const nbNumerator = 0.5 * nbWord1 * nbWord2;

  // --- 5. RANDOM FOREST STATES ---
  const [rfTrees, setRfTrees] = useState(3);
  const treeVariations = [
    { rotate: 10, color: '#3b82f6', name: 'Cây 1' },
    { rotate: -15, color: '#10b981', name: 'Cây 2' },
    { rotate: 45, color: '#f59e0b', name: 'Cây 3' },
    { rotate: -30, color: '#ef4444', name: 'Cây 4' },
    { rotate: 5, color: '#8b5cf6', name: 'Cây 5' },
  ];

  // --- 6. PHOBERT STATES ---
  const [pbWordIdx, setPbWordIdx] = useState(0);
  const pbWords = ["Ngân", "hàng", "trung", "ương", "vừa", "tăng", "lãi", "suất"];
  const pbAttention = [
    [1.0, 0.8, 0.4, 0.3, 0.1, 0.2, 0.6, 0.4], // Ngân
    [0.8, 1.0, 0.3, 0.2, 0.1, 0.1, 0.5, 0.3], // hàng
    [0.4, 0.3, 1.0, 0.9, 0.2, 0.3, 0.7, 0.5], // trung
    [0.3, 0.2, 0.9, 1.0, 0.1, 0.2, 0.6, 0.4], // ương
    [0.1, 0.1, 0.2, 0.1, 1.0, 0.8, 0.3, 0.2], // vừa
    [0.2, 0.1, 0.3, 0.2, 0.8, 1.0, 0.9, 0.7], // tăng
    [0.6, 0.5, 0.7, 0.6, 0.3, 0.9, 1.0, 0.8], // lãi
    [0.4, 0.3, 0.5, 0.4, 0.2, 0.7, 0.8, 1.0], // suất
  ];

  return (
    <div className="space-y-32 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <section>
        <h1 className="text-6xl font-black mb-6 text-slate-900 tracking-tight">3. 8 Thuật toán <span className="text-blue-600">Phân loại AI</span></h1>
        <p className="text-xl text-slate-500 leading-relaxed max-w-3xl font-medium">
          Mỗi thuật toán là một "góc nhìn" khác nhau về dữ liệu. Dưới đây là mô phỏng động chi tiết về 8 chiến binh AI trong trận chiến chống tin giả.
        </p>
      </section>

      {/* --- 1. LOGISTIC REGRESSION --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-blue-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-blue-200">1</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">Hồi quy Logistic</h2>
            <p className="text-blue-600 font-bold uppercase text-xs tracking-widest mt-1">Phân loại dựa trên xác suất</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="space-y-8">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-blue-600 rounded-full"></span> 🟢 Khái niệm
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Hồi quy Logistic là thuật toán dự đoán xác suất một đối tượng thuộc về một nhóm (Thật/Giả). Thay vì vẽ đường thẳng, nó sử dụng hàm đặc biệt (Sigmoid) để "nén" mọi điểm số vào khoảng [0, 1].
              </p>
            </section>

            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-blue-600 rounded-full"></span> 📐 Công thức
              </h4>
              <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
                <p className="text-center text-xl font-serif italic mb-4">P = 1 / (1 + e<sup>-(W*X + B)</sup>)</p>
                <div className="text-[11px] text-slate-500 leading-relaxed space-y-1">
                  <p>• <strong>W (Weight)</strong>: Độ ám ảnh của Mô hình với từ khóa này.</p>
                  <p>• <strong>X (Feature)</strong>: Số lần từ khóa xuất hiện trong bài.</p>
                  <p>• <strong>B (Bias)</strong>: Định kiến ban đầu của mô hình.</p>
                </div>
              </div>
            </section>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Đơn giản, tốc độ cực nhanh, kết quả trả về là xác suất rõ ràng dễ diễn giải.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Kém hiệu quả khi dữ liệu phức tạp không thể chia bằng đường thẳng.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center">
             <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-8">Mô phỏng hàm Sigmoid</h5>
             <div className="relative w-full h-48 border-b border-l border-slate-700 mb-10">
               <svg className="absolute inset-0 w-full h-full overflow-visible">
                 <path d={Array.from({length: 100}).map((_, i) => {
                   const xVal = (i / 50 - 1) * 6;
                   const yVal = sigmoid(xVal * lrW + lrB);
                   return `${i === 0 ? 'M' : 'L'} ${(i/100)*100}% ${192 - yVal * 192}`;
                 }).join(' ')} fill="none" stroke="#3b82f6" strokeWidth="4" />
                 <line x1="0" y1="50%" x2="100%" y2="50%" stroke="#1e293b" strokeDasharray="4" />
                 <circle cx={`${((lrX + 6) / 12) * 100}%`} cy={`${192 - lrP * 192}`} r="8" fill="#fbbf24" className="animate-pulse shadow-[0_0_20px_#fbbf24]" />
               </svg>
             </div>
             
             <div className="w-full space-y-6">
               <div className="flex flex-col gap-2">
                 <div className="flex justify-between text-[10px] font-black text-slate-400"><span>ĐỘ QUAN TRỌNG TỪ KHÓA (W)</span> <span>{lrW.toFixed(1)}</span></div>
                 <input type="range" min="0.5" max="5" step="0.1" value={lrW} onChange={(e) => setLrW(parseFloat(e.target.value))} className="w-full accent-blue-600" />
               </div>
               <div className="flex flex-col gap-2">
                 <div className="flex justify-between text-[10px] font-black text-slate-400"><span>ĐỊNH KIẾN (B)</span> <span>{lrB.toFixed(1)}</span></div>
                 <input type="range" min="-3" max="3" step="0.1" value={lrB} onChange={(e) => setLrB(parseFloat(e.target.value))} className="w-full accent-blue-600" />
               </div>
               <div className="flex flex-col gap-2">
                 <div className="flex justify-between text-[10px] font-black text-slate-400"><span>SỰ XUẤT HIỆN TỪ KHÓA (X)</span> <span>{lrX.toFixed(1)}</span></div>
                 <input type="range" min="-5" max="5" step="0.1" value={lrX} onChange={(e) => setLrX(parseFloat(e.target.value))} className="w-full accent-blue-600" />
               </div>
             </div>
             
             <div className="mt-10 p-4 w-full bg-white/5 border border-white/10 rounded-2xl text-center">
               <div className="text-4xl font-black text-yellow-400">{(lrP * 100).toFixed(1)}%</div>
               <p className="text-[10px] font-black text-slate-500 uppercase mt-2 tracking-widest">{lrP >= 0.5 ? "Kết luận: TIN GIẢ" : "Kết luận: TIN THẬT"}</p>
             </div>
          </div>
        </div>
      </div>

      {/* --- 2. SVM --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-purple-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-purple-200">2</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">Máy vectơ hỗ trợ (SVM)</h2>
            <p className="text-purple-600 font-bold uppercase text-xs tracking-widest mt-1">Tối ưu hóa khoảng cách lề</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center order-2 lg:order-1">
            <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-8 text-center">Mô phỏng Con đường & Lề đường</h5>
            <div className="relative w-64 h-64 bg-slate-800 rounded-[2rem] border border-slate-700 overflow-hidden">
               {/* Các điểm dữ liệu bình thường */}
               <div className="absolute top-8 left-8 w-3 h-3 bg-blue-500/30 rounded-full"></div>
               <div className="absolute bottom-8 right-8 w-3 h-3 bg-red-500/30 rounded-full"></div>
               
               {/* Vectơ hỗ trợ - Các điểm quan trọng nhất */}
               <div className="absolute top-36 left-28 w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-[0_0_15px_#3b82f6]"></div>
               <div className="absolute bottom-36 right-28 w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-[0_0_15px_#ef4444]"></div>
               
               <div className="absolute top-0 left-0 w-[200%] h-1 bg-white rotate-45 origin-top-left"></div>
               <div className="absolute top-0 left-0 w-[200%] h-0.5 bg-yellow-400 rotate-45 origin-top-left transition-all duration-300" style={{ transform: `translateY(-${marginSize}px) rotate(45deg)` }}></div>
               <div className="absolute top-0 left-0 w-[200%] h-0.5 bg-yellow-400 rotate-45 origin-top-left transition-all duration-300" style={{ transform: `translateY(${marginSize}px) rotate(45deg)` }}></div>
            </div>
            
            <div className="w-full mt-10 space-y-4">
               <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase"><span>Độ rộng con đường (Lề)</span> <span>{(2/svmW).toFixed(2)}</span></div>
               <input type="range" min="0.5" max="3" step="0.1" value={svmW} onChange={(e) => setSvmW(parseFloat(e.target.value))} className="w-full accent-purple-500" />
            </div>
            <div className="mt-6 p-4 bg-purple-900/30 rounded-2xl border border-purple-500/20 text-left">
               <p className="text-xs text-purple-200 font-bold mb-2">💡 Tại sao Lề rộng lại chống được nhiễu?</p>
               <p className="text-[10px] text-purple-300 leading-relaxed">
                  Lề (Margin) giống như <strong>"vùng đệm an toàn"</strong>. Nếu biên giới quá chật hẹp, chỉ cần một bài báo Tin Thật bị viết hơi giống văn phong Tin Giả (nhiễu), nó sẽ dễ dàng bị nhận nhầm. <br/><br/>
                  Một lề rộng giúp mô hình có sự "khoan dung" nhất định với các điểm dữ liệu bất thường, từ đó đưa ra quyết định vững chắc hơn khi gặp các bài báo mới trong thực tế (tránh Overfitting).
               </p>
            </div>
          </div>

          <div className="space-y-8 order-1 lg:order-2">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-purple-600 rounded-full"></span> 🟢 Khái niệm
              </h4>
              <p className="text-slate-600 leading-relaxed">
                SVM tìm một "siêu phẳng" (hyperplane) để chia dữ liệu sao cho khoảng cách từ các điểm gần nhất (Vectơ hỗ trợ) đến ranh giới đó là **lớn nhất có thể**.
              </p>
            </section>

            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-purple-600 rounded-full"></span> 📐 Công thức & Ý nghĩa
              </h4>
              <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100 space-y-2">
                <p className="font-bold text-slate-700">Siêu phẳng: wᵀx + b = 0</p>
                <p className="font-bold text-slate-700">Khoảng cách lề: 2 / ||w||</p>
                <div className="text-[11px] text-slate-500 italic mt-2">
                  • <strong>Vectơ hỗ trợ</strong>: Chỉ những điểm nằm sát rào chắn (có viền trắng) mới quyết định đường biên. Những điểm ở xa hoàn toàn vô tác dụng.
                </div>
              </div>
            </section>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Rất mạnh khi dữ liệu có số chiều lớn (như các vectơ TF-IDF), tìm ra ranh giới dứt khoát nhất.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Rất chậm và ngốn phần cứng nếu tập dữ liệu quá khổng lồ.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* --- 3. DECISION TREE --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-green-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-green-200">3</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">Cây quyết định (Decision Tree)</h2>
            <p className="text-green-600 font-bold uppercase text-xs tracking-widest mt-1">Lọc sạch dữ liệu bằng câu hỏi</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="space-y-8">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-green-600 rounded-full"></span> 🟢 Độ hỗn loạn (Entropy)
              </h4>
              <p className="text-slate-600 leading-relaxed text-sm">
                Cây sẽ hỏi các câu (VD: Bài có chữ "Tuyệt mật"?). Mục đích là đưa dữ liệu về các nhánh nhỏ hơn sao cho mỗi nhánh **chỉ chứa toàn bộ 1 loại tin** (Sạch 100%, Entropy = 0).
              </p>
            </section>

            <section className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                 <div className="p-4 bg-red-50 rounded-2xl border border-red-100 text-center">
                    <div className="text-2xl font-black text-red-600">Entropy = 1.0</div>
                    <p className="text-[10px] font-bold text-slate-500 uppercase mt-1">Lẫn lộn 50/50 (Tệ)</p>
                 </div>
                 <div className="p-4 bg-green-50 rounded-2xl border border-green-100 text-center">
                    <div className="text-2xl font-black text-green-600">Entropy = 0.0</div>
                    <p className="text-[10px] font-bold text-slate-500 uppercase mt-1">Sạch (Hoàn hảo)</p>
                 </div>
              </div>
            </section>
            
            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Dễ hiểu như cách bộ não người suy luận đúng/sai, không cần chuẩn hóa dữ liệu nhiều trước khi nạp.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Dễ bị "Overfitting" (học vẹt), thuộc lòng data cũ nhưng sai lệch hẳn với data mới.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center">
            <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-8 text-center">Mô phỏng "Mặt Cắt" Phân Loại</h5>
            <div className="relative w-full h-40 bg-slate-800 rounded-3xl overflow-hidden flex items-center border border-slate-700 shadow-inner">
               <div className="h-full bg-blue-500/10 border-r-4 border-green-400 flex flex-wrap p-4 content-center transition-all duration-300 relative" style={{ width: `${dtSplit}%` }}>
                 <div className="absolute top-2 left-2 text-[8px] font-black text-slate-500 uppercase">Nhánh Trái (Được Lọc)</div>
                 {Array.from({length: leftCount}).map((_, i) => (
                   <div key={i} className={`w-3 h-3 m-1 rounded-full ${i < leftRed ? 'bg-red-500 shadow-[0_0_10px_#ef4444]' : 'bg-blue-500 shadow-[0_0_10px_#3b82f6]'}`}></div>
                 ))}
               </div>
               <div className="flex-1 h-full flex flex-wrap p-4 content-center relative">
                  <div className="absolute top-2 left-2 text-[8px] font-black text-slate-500 uppercase">Nhánh Phải</div>
                  {Array.from({length: rightCount}).map((_, i) => (
                   <div key={i} className={`w-3 h-3 m-1 rounded-full ${i < rightRed ? 'bg-red-500' : 'bg-blue-500'}`}></div>
                 ))}
               </div>
            </div>
            
            <div className="w-full mt-10 space-y-6">
               <div className="flex flex-col gap-2">
                 <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase"><span>Điều chỉnh Nhát cắt Nhánh Trái</span> <span>{dtSplit}% data</span></div>
                 <input type="range" min="10" max="90" value={dtSplit} onChange={(e) => setDtSplit(parseInt(e.target.value))} className="w-full accent-green-500" />
               </div>
               
               <div className="p-6 bg-white/5 rounded-3xl border border-white/10 w-full transition-colors">
                  <div className="flex justify-between items-center mb-4">
                     <span className="text-[10px] font-black text-slate-400 uppercase">Độ hỗn loạn Nhánh Trái (Entropy)</span>
                     <span className={`text-xl font-black transition-colors ${entropyLeft === 0 ? 'text-green-400' : 'text-yellow-400'}`}>{entropyLeft.toFixed(3)}</span>
                  </div>
                  <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                     <div className="h-full bg-green-400 transition-all duration-300" style={{ width: `${(1 - entropyLeft) * 100}%` }}></div>
                  </div>
                  <p className="text-[9px] text-slate-500 mt-3 text-center italic">Cố gắng điều chỉnh để thanh màu xanh lấp đầy (Entropy về 0)</p>
               </div>
            </div>
          </div>
        </div>
      </div>

      {/* --- 4. NAIVE BAYES --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-orange-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-orange-200">4</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">Naive Bayes</h2>
            <p className="text-orange-600 font-bold uppercase text-xs tracking-widest mt-1">Xác suất thống kê "Ngây thơ"</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col order-2 lg:order-1">
             <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-10 text-center">Cách Máy Tính Nhân Xác Suất</h5>
             <div className="text-center space-y-8">
                <div className="flex items-center justify-around">
                   <div className="text-xl font-bold text-orange-400">0.5 <div className="text-[8px] text-slate-500 uppercase">Cơ sở</div></div>
                   <div className="text-2xl text-slate-700">×</div>
                   <div className="text-xl font-bold text-orange-400">{nbWord1.toFixed(2)} <div className="text-[8px] text-slate-500 uppercase">Từ khóa A</div></div>
                   <div className="text-2xl text-slate-700">×</div>
                   <div className="text-xl font-bold text-orange-400">{nbWord2.toFixed(2)} <div className="text-[8px] text-slate-500 uppercase">Từ khóa B</div></div>
                </div>
                <div className="text-5xl font-black text-yellow-400 pt-6 border-t border-slate-800">{nbNumerator.toFixed(4)}</div>
             </div>
             
             <div className="mt-12 space-y-6">
               <div className="flex flex-col gap-2">
                 <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase"><span>Độ nguy hiểm Từ khóa A</span> <span>{(nbWord1 * 100).toFixed(0)}%</span></div>
                 <input type="range" min="0.1" max="0.9" step="0.01" value={nbWord1} onChange={(e) => setNbWord1(parseFloat(e.target.value))} className="w-full accent-orange-500" />
               </div>
               <div className="flex flex-col gap-2">
                 <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase"><span>Độ nguy hiểm Từ khóa B</span> <span>{(nbWord2 * 100).toFixed(0)}%</span></div>
                 <input type="range" min="0.1" max="0.9" step="0.01" value={nbWord2} onChange={(e) => setNbWord2(parseFloat(e.target.value))} className="w-full accent-orange-500" />
               </div>
             </div>
          </div>

          <div className="space-y-8 order-1 lg:order-2">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-orange-600 rounded-full"></span> 🟢 Tại sao lại "Ngây thơ"?
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Thuật toán giả định rằng tất cả từ khóa xuất hiện hoàn toàn **độc lập** với nhau (ví dụ: chữ "miễn phí" không liên quan gì đến chữ "trúng thưởng"). Nó chỉ đơn giản nhân mọi xác suất lại với nhau.
              </p>
            </section>

            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-orange-600 rounded-full"></span> 📐 Định lý Bayes
              </h4>
              <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100 text-center font-serif italic text-xl">
                P(Giả|Text) = [P(Text|Giả) * P(Giả)] / P(Text)
              </div>
              <p className="text-xs text-slate-500 font-bold leading-relaxed mt-4">
                Sức mạnh lớn nhất của Naive Bayes là: Cực kỳ nhanh, hiệu quả cực tốt với bài toán Lọc Spam Email và Tin tức.
              </p>
            </section>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Cực kì nhanh (nhiều khi chỉ cần vài phép nhân vi mô), sinh ra để tối ưu mảng phân loại văn bản.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Giả định "ngây thơ" rằng các từ hoàn toàn bị cô lập, bỏ qua hoàn toàn ngữ cảnh của các chuỗi từ.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* --- 5. RANDOM FOREST --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-red-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-red-200">5</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">Rừng ngẫu nhiên (Random Forest)</h2>
            <p className="text-red-600 font-bold uppercase text-xs tracking-widest mt-1">Sức mạnh của bầu chọn đa số</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="space-y-8">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-red-600 rounded-full"></span> 🟢 Khái niệm Ensemble Learning
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Khắc phục nhược điểm "học vẹt" của 1 Cây quyết định duy nhất. Rừng ngẫu nhiên xây dựng hàng trăm cây quyết định khác nhau. Mỗi cây đưa ra dự đoán của mình, và kết quả sẽ được lấy theo số đông (Voting).
              </p>
            </section>

            <div className="p-6 bg-red-600 rounded-3xl text-white shadow-lg relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
              <h5 className="text-[10px] font-black uppercase mb-2 text-red-200 tracking-widest">Triết lý cốt lõi</h5>
              <p className="text-base font-bold italic">"Một cây có thể sai hoặc điên rồ, nhưng cả một khu rừng thì rất khó bị đánh lừa bởi dữ liệu nhiễu."</p>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Độ chính xác khổng lồ, xử lý triệt để căn bệnh "học vẹt" (Overfitting) bằng bầu chọn đa số.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Hoạt động như hộp đen khổng lồ, khó để con người giải thích vì sao AI lại quyết định ra được kết quả đó.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center">
            <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-10 text-center">Sự đa dạng của các ranh giới</h5>
            
            <div className="relative w-64 h-64 bg-slate-800 rounded-[2.5rem] border border-slate-700 overflow-hidden flex items-center justify-center shadow-inner">
              {treeVariations.slice(0, rfTrees).map((t, i) => (
                <div key={i} className="absolute w-full h-1 opacity-50 transition-all duration-500" style={{ backgroundColor: t.color, transform: `rotate(${t.rotate}deg)` }}></div>
              ))}
              <div className="z-10 bg-slate-900 px-4 py-2 rounded-xl border border-white/20 text-[10px] font-black shadow-2xl">BIÊN GIỚI TỔNG HỢP</div>
            </div>
            
            <div className="w-full mt-10 space-y-4">
               <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase"><span>Số lượng Cây tham gia bầu chọn</span> <span>{rfTrees} Trees</span></div>
               <input type="range" min="1" max="5" value={rfTrees} onChange={(e) => setRfTrees(parseInt(e.target.value))} className="w-full accent-red-600" />
            </div>
          </div>
        </div>
      </div>

      {/* --- 6. PHOBERT --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-cyan-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-cyan-200">6</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">PhoBERT (Transformer)</h2>
            <p className="text-cyan-600 font-bold uppercase text-xs tracking-widest mt-1">Cơ chế Chú ý (Self-Attention)</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="space-y-8">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-cyan-600 rounded-full"></span> 🟢 Đỉnh cao công nghệ NLP
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Khác với các mô hình trên, PhoBERT không chỉ "đếm từ". Nó "đọc" toàn bộ câu để hiểu ngữ cảnh. Chữ <strong>"đường"</strong> trong "đường ăn" sẽ được hiểu khác hoàn toàn với "đường đi" nhờ cơ chế Attention.
              </p>
            </section>

            <div className="p-6 bg-cyan-50 rounded-3xl border border-cyan-100">
               <h5 className="text-[10px] font-black text-cyan-600 uppercase mb-4 tracking-widest">Mô phỏng Self-Attention</h5>
               <div className="flex flex-wrap gap-2 mb-6">
                  {pbWords.map((word, i) => (
                    <button 
                      key={i} 
                      onMouseEnter={() => setPbWordIdx(i)}
                      className={`px-3 py-1.5 rounded-xl text-xs font-black transition-all border-2 ${
                        pbWordIdx === i ? 'bg-cyan-600 text-white border-cyan-600 scale-110' : 'bg-white text-slate-400 border-slate-100 hover:border-cyan-300'
                      }`}
                    >
                      {word}
                    </button>
                  ))}
               </div>
               <div className="space-y-2">
                  <p className="text-[10px] font-black text-slate-400 uppercase">Mối liên kết với từ khác:</p>
                  <div className="flex items-end gap-1 h-12">
                     {pbAttention[pbWordIdx].map((val, i) => (
                       <div 
                        key={i} 
                        className="bg-cyan-600 rounded-t-sm transition-all duration-300" 
                        style={{ height: `${val * 100}%`, width: '12%', opacity: 0.3 + val * 0.7 }}
                        title={pbWords[i]}
                       ></div>
                     ))}
                  </div>
                  <p className="text-[9px] text-slate-400 italic mt-2 text-center">Di chuột qua từng từ ở trên để thấy AI "chú ý" vào các từ liên quan</p>
               </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Hiểu ngữ nghĩa sâu sắc, tiếng lóng, ngữ pháp tiếng Việt tinh vi. Chính xác nhất hiện nay.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Rất nặng (cần hàng GB RAM), tốc độ dự đoán chậm và yêu cầu máy tính có GPU mạnh.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center">
            <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-10 text-center">Kiến trúc Transformer</h5>
            
            <div className="w-full space-y-4">
               {[
                 { name: 'Classification Head', color: 'bg-red-500', desc: 'Đưa ra kết luận: Thật / Giả' },
                 { name: 'Pooling Layer', color: 'bg-yellow-500', desc: 'Nén toàn bộ ý nghĩa vào 1 vectơ' },
                 { name: '12 Layers Transformer', color: 'bg-cyan-600', desc: 'Hàng triệu tham số phân tích ngữ nghĩa' },
                 { name: 'Word Embedding', color: 'bg-indigo-600', desc: 'Chuyển chữ thành tọa độ không gian' },
               ].map((layer, i) => (
                 <div key={i} className={`p-4 rounded-2xl border border-white/10 ${layer.color} bg-opacity-20 flex flex-col items-center text-center animate-pulse`} style={{ animationDelay: `${i * 0.2}s` }}>
                    <div className="text-[10px] font-black uppercase mb-1">{layer.name}</div>
                    <div className="text-[8px] text-white/60 font-medium uppercase tracking-tighter">{layer.desc}</div>
                 </div>
               ))}
               <div className="text-center pt-4">
                  <div className="text-cyan-400 text-2xl font-black">↑ ↑ ↑</div>
                  <div className="text-[10px] font-black text-slate-500 uppercase mt-1 tracking-[0.3em]">VĂN BẢN ĐẦU VÀO</div>
               </div>
            </div>
          </div>
        </div>
      </div>

      {/* --- 7. VIBERT --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-blue-800 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-blue-200">7</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">ViBERT (FPTAI)</h2>
            <p className="text-blue-800 font-bold uppercase text-xs tracking-widest mt-1">Mô hình BERT tối ưu cho Tiếng Việt</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center order-2 lg:order-1">
             <div className="w-full space-y-4">
                <div className="p-6 bg-blue-900/30 rounded-3xl border border-blue-500/20">
                   <h5 className="text-[10px] font-black text-blue-400 uppercase mb-4 tracking-widest text-center">Đào tạo qua MLM</h5>
                   <div className="text-center font-bold text-lg mb-4">
                      Hôm nay trời rất <span className="bg-blue-600 px-2 rounded">[MASK]</span>
                   </div>
                   <div className="flex justify-center gap-2">
                      <div className="px-3 py-1 bg-green-600/20 text-green-400 rounded-lg text-xs font-black">đẹp (85%)</div>
                      <div className="px-3 py-1 bg-yellow-600/20 text-yellow-400 rounded-lg text-xs font-black">xấu (10%)</div>
                      <div className="px-3 py-1 bg-red-600/20 text-red-400 rounded-lg text-xs font-black">nóng (5%)</div>
                   </div>
                </div>
                <p className="text-[10px] text-slate-400 leading-relaxed text-center italic mt-4">
                   ViBERT học ngôn ngữ bằng cách "điền vào chỗ trống", giúp nó nắm bắt quy luật ngữ pháp và từ vựng tiếng Việt một cách tự nhiên nhất.
                </p>
             </div>
          </div>

          <div className="space-y-8 order-1 lg:order-2">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-blue-800 rounded-full"></span> 🟢 Khái niệm
              </h4>
              <p className="text-slate-600 leading-relaxed">
                ViBERT là một biến thể của BERT, được FPTAI huấn luyện với lượng dữ liệu tiếng Việt khổng lồ từ các trang tin tức chính thống. Nó kế thừa đầy đủ sức mạnh của Transformer nhưng tập trung vào sự đa dạng của từ vựng Việt Nam.
              </p>
            </section>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Hiệu năng phân loại sắc thái và ý định cực tốt nhờ kho từ vựng tin tức phong phú.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Cần tài nguyên máy tính cao, tốc độ xử lý chậm hơn các mô hình truyền thống.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* --- 8. SBERT --- */}
      <div className="space-y-12">
        <div className="flex items-center gap-6">
          <div className="w-16 h-16 bg-indigo-600 rounded-[2rem] flex items-center justify-center text-white text-3xl font-black shadow-xl shadow-indigo-200">8</div>
          <div>
            <h2 className="text-4xl font-black text-slate-900">Vietnamese-SBERT</h2>
            <p className="text-indigo-600 font-bold uppercase text-xs tracking-widest mt-1">So sánh Ngữ nghĩa Câu văn</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
          <div className="space-y-8">
            <section className="space-y-4">
              <h4 className="text-lg font-black text-slate-900 flex items-center gap-2">
                <span className="w-2 h-6 bg-indigo-600 rounded-full"></span> 🟢 Khái niệm "Cặp song sinh"
              </h4>
              <p className="text-slate-600 leading-relaxed">
                Khác với BERT truyền thống, SBERT được huấn luyện với cấu trúc Siamese (Cặp song sinh) để đưa toàn bộ câu văn về một tọa độ duy nhất trong không gian đa chiều. Nhờ đó, máy tính có thể so sánh độ giống nhau giữa 2 bài báo một cách tức thời.
              </p>
            </section>

            <div className="bg-indigo-50 p-6 rounded-3xl border border-indigo-100 space-y-4">
               <h5 className="text-[10px] font-black text-indigo-600 uppercase tracking-widest">Kiến trúc Siamese Network</h5>
               <div className="flex justify-between items-center px-4">
                  <div className="bg-white p-3 rounded-xl border border-indigo-200 text-[10px] font-bold shadow-sm">Câu văn A</div>
                  <div className="text-indigo-300 font-black">➔ [ BERT ] ➔</div>
                  <div className="bg-indigo-600 text-white p-3 rounded-xl text-[10px] font-bold shadow-lg">Vectơ 768 chiều</div>
               </div>
               <div className="flex justify-between items-center px-4">
                  <div className="bg-white p-3 rounded-xl border border-indigo-200 text-[10px] font-bold shadow-sm">Câu văn B</div>
                  <div className="text-indigo-300 font-black">➔ [ BERT ] ➔</div>
                  <div className="bg-indigo-600 text-white p-3 rounded-xl text-[10px] font-bold shadow-lg">Vectơ 768 chiều</div>
               </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4">
              <div className="p-5 bg-green-50 rounded-3xl border border-green-100">
                <h5 className="text-[10px] font-black text-green-600 uppercase mb-2">Ưu điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Đỉnh cao trong việc tìm kiếm thông tin và nhận biết các bài báo có nội dung tương tự nhau.</p>
              </div>
              <div className="p-5 bg-red-50 rounded-3xl border border-red-100">
                <h5 className="text-[10px] font-black text-red-600 uppercase mb-2">Nhược điểm</h5>
                <p className="text-xs font-bold text-slate-700 leading-relaxed">Phức tạp hơn trong việc tinh chỉnh cho các bài toán phân loại nhãn nhị phân.</p>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 p-10 rounded-[3rem] text-white shadow-2xl flex flex-col items-center">
             <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-10 text-center">Bản đồ Ngữ nghĩa (Vector Space)</h5>
             <div className="relative w-64 h-64 border border-slate-700 rounded-full overflow-hidden flex items-center justify-center bg-[radial-gradient(circle_at_center,#1e293b_0%,#0f172a_100%)]">
                <div className="absolute w-full h-px bg-slate-800"></div>
                <div className="absolute h-full w-px bg-slate-800"></div>
                
                {/* Các cụm tin tức */}
                <div className="absolute top-1/4 left-1/4">
                   <div className="w-16 h-16 bg-blue-500/20 rounded-full blur-xl"></div>
                   <div className="w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_10px_#3b82f6]"></div>
                   <div className="text-[8px] text-blue-300 font-bold mt-1">Cụm tin Thật</div>
                </div>
                
                <div className="absolute bottom-1/4 right-1/4">
                   <div className="w-16 h-16 bg-red-500/20 rounded-full blur-xl"></div>
                   <div className="w-2 h-2 bg-red-500 rounded-full shadow-[0_0_10px_#ef4444]"></div>
                   <div className="text-[8px] text-red-300 font-bold mt-1">Cụm tin Giả</div>
                </div>
                
                {/* Tin mới đang so sánh */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-bounce">
                   <div className="w-3 h-3 bg-yellow-400 rounded-full border-2 border-white shadow-[0_0_20px_#fbbf24]"></div>
                </div>
             </div>
             <p className="text-[10px] text-slate-500 mt-8 text-center uppercase tracking-widest font-black">AI đang đo khoảng cách để phân loại</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chapter3Models;
