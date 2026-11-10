

const Chapter6Pipeline = () => {
  const steps = [
    {
       num: 1,
       icon: "📝",
       title: "Thu thập & Nhập liệu (Input)",
       desc: "Thu thập văn bản tin tức từ các nguồn (báo chí, mạng xã hội). Dữ liệu ở dạng thô, chứa nhiều nhiễu và từ ngữ không cần thiết.",
       color: "bg-slate-500",
       glow: "shadow-slate-500/50"
    },
    {
       num: 2,
       icon: "🧼",
       title: "Tiền xử lý (Preprocessing)",
       desc: "Làm sạch văn bản bằng Regex, xóa Stop-words, đưa về chữ thường. Tách từ bằng AI (Underthesea) để giữ nguyên cấu trúc các từ ghép Tiếng Việt.",
       color: "bg-sky-500",
       glow: "shadow-sky-500/50"
    },
    {
       num: 3,
       icon: "🔢",
       title: "Vectơ hóa (Vectorization)",
       desc: "Biến đổi văn bản chữ thành các ma trận số học (TF-IDF hoặc Doc2Vec) để máy tính có thể hiểu được trọng số và ngữ nghĩa của câu.",
       color: "bg-indigo-500",
       glow: "shadow-indigo-500/50"
    },
    {
       num: 4,
       icon: "🧠",
       title: "Mô hình Phân loại (Model)",
       desc: "Đưa các Vectơ số vào các thuật toán Machine Learning (Logistic, SVM, Random Forest...) để tính toán xác suất tin giả dựa trên dữ liệu đã học.",
       color: "bg-rose-500",
       glow: "shadow-rose-500/50"
    },
    {
       num: 5,
       icon: "🎯",
       title: "Kết quả (Output)",
       desc: "Mô hình đưa ra quyết định cuối cùng và cung cấp tỉ lệ phần trăm đáng tin cậy. Đánh giá lại bằng các chỉ số Metrics và Confusion Matrix.",
       color: "bg-emerald-500",
       glow: "shadow-emerald-500/50"
    }
  ];

  return (
    <div className="space-y-32 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <section>
        <h1 className="text-6xl font-black mb-6 text-slate-900 tracking-tight">6. Luồng <span className="text-blue-600">Xử Lý Hệ Thống</span></h1>
        <p className="text-xl text-slate-500 leading-relaxed max-w-3xl font-medium">
          Một cái nhìn toàn cảnh về hành trình của dữ liệu: Từ khi là một đoạn văn bản thô sơ cho đến khi trở thành một bản án "Thật" hay "Giả" đanh thép của AI.
        </p>
      </section>

      <div className="bg-slate-900 p-8 md:p-16 rounded-[4rem] shadow-2xl relative overflow-hidden">
         {/* Background Elements */}
         <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl -mt-20 -mr-20"></div>
         <div className="absolute bottom-0 left-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl -mb-20 -ml-20"></div>
         
         <div className="relative z-10 w-full max-w-4xl mx-auto">
            {/* Đường nối chung (Timeline line) */}
            <div className="absolute top-0 bottom-0 left-12 md:left-1/2 w-1 bg-slate-800 -translate-x-1/2 rounded-full hidden sm:block"></div>

            <div className="space-y-12">
               {steps.map((step, index) => (
                  <div key={index} className={`flex flex-col sm:flex-row items-center gap-8 ${index % 2 === 0 ? 'sm:flex-row-reverse' : ''}`}>
                     
                     {/* Nội dung Bước */}
                     <div className={`w-full sm:w-1/2 p-6 md:p-8 bg-slate-800/80 rounded-3xl border border-slate-700/50 backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:bg-slate-800 relative group`}>
                        <div className={`absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-10 transition-opacity rounded-3xl ${step.color === 'bg-sky-500' ? 'from-sky-500 to-transparent' : step.color === 'bg-indigo-500' ? 'from-indigo-500 to-transparent' : step.color === 'bg-rose-500' ? 'from-rose-500 to-transparent' : step.color === 'bg-emerald-500' ? 'from-emerald-500 to-transparent' : 'from-slate-500 to-transparent'}`}></div>
                        <h4 className="text-xl font-black text-white mb-3 flex items-center gap-3">
                           <span className="text-2xl">{step.icon}</span>
                           <span>{step.title}</span>
                        </h4>
                        <p className="text-sm text-slate-400 leading-relaxed font-medium">
                           {step.desc}
                        </p>
                     </div>

                     {/* Vòng tròn Số (Nút) */}
                     <div className="hidden sm:flex relative z-10 shrink-0 w-24 h-24 bg-slate-900 border-4 border-slate-800 rounded-full items-center justify-center">
                        <div className={`w-16 h-16 ${step.color} rounded-full flex items-center justify-center text-white text-2xl font-black shadow-lg ${step.glow}`}>
                           {step.num}
                        </div>
                     </div>

                     {/* Đệm chèn giữ 레이아웃 (Layout spacer) */}
                     <div className="hidden sm:block w-1/2"></div>
                  </div>
               ))}
            </div>
         </div>
         
         {/* Footer Mô Phỏng */}
         <div className="mt-20 border-t border-slate-800 pt-10 text-center relative z-10">
            <h3 className="text-sm font-black text-slate-500 tracking-widest uppercase mb-8">Kiến trúc Tổng thể Model (Architecture)</h3>
            <div className="flex flex-wrap justify-center items-center gap-4 text-xs font-mono">
               <span className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg border border-slate-700">Raw Data</span>
               <span className="text-slate-500">→</span>
               <span className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg border border-slate-700">Underthesea_Tokenizer</span>
               <span className="text-slate-500">→</span>
               <span className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg border border-slate-700">TfidfVectorizer</span>
               <span className="text-slate-500">→</span>
               <span className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg border border-slate-700">LogisticRegression.predict()</span>
               <span className="text-slate-500">→</span>
               <span className="px-4 py-2 bg-emerald-900/50 text-emerald-400 font-bold rounded-lg border border-emerald-500/50">[1: FAKE, 0: REAL]</span>
            </div>
         </div>
      </div>
    </div>
  );
};

export default Chapter6Pipeline;
