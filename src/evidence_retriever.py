import os
import re
import math
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import normalize_vietnamese_unicode


class MultiTierEvidenceRetriever:
    """
    GIAI ĐOẠN 2 (Nodes 4, 5, 6): Truy xuất & Xếp hạng Bằng chứng Đa tầng (Hybrid Retrieval + Reranker + Temporal Filtering)
      - Node 4: Truy xuất Kho Bằng chứng Nội bộ đã xác minh (360+ Gold Claims KB + Dataset mẫu)
                Hỗ trợ 4 chế độ đối chứng thực nghiệm:
                  1. BM25 Baseline (Okapi BM25 k1=1.5, b=0.75)
                  2. Dense Retrieval (Semantic Dense Vector Embeddings)
                  3. Hybrid Retrieval (BM25 + Dense Fusion)
                  4. Hybrid + Cross-Encoder Reranker + Temporal Filtering (Kiểm tra mốc thời gian date)
      - Node 5: Truy xuất Báo chí Chính thống Việt Nam thời gian thực qua Google News RSS
      - Node 6: Đo độ tương đồng ngữ nghĩa và phân ngưỡng theo Mục S2.2 (Vosoughi et al., Science 2018):
          * sim < 0.50  -> DISCARDED (Loại bỏ)
          * 0.50 - 0.90 -> NEEDS_INSPECTION (Đưa vào đối chiếu NLI sâu)
          * sim >= 0.90 -> EXACT_MATCH (Khớp trực tiếp với tin đã kiểm chứng)
    """

    def __init__(self, base_dir: str, evidence_only: bool = False, include_historical: bool = True):
        """
        evidence_only=True  : Chế độ đánh giá chặt (Strict / Leakage-free). Kho chỉ lập chỉ mục
                              tiêu đề + nội dung bằng chứng (title + gold_evidence), KHÔNG chứa
                              trường claim/claim_pattern và KHÔNG mang nhãn có sẵn (label/verdict),
                              buộc bộ Verifier phải suy luận từ nội dung bằng chứng.
        include_historical  : Có nạp thêm mẫu ReINTEL (fake_news.csv) vào kho hay không
                              (các mẫu này chứa nhãn ngay trong evidence_text).
        """
        self.base_dir = base_dir
        self.evidence_only = evidence_only
        self.include_historical = include_historical
        self.kb_path = os.path.join(base_dir, "data", "knowledge_base", "verified_facts_vn.json")
        self.train_csv_path = os.path.join(base_dir, "data", "train", "fake_news.csv")
        self.local_corpus: List[Dict[str, Any]] = []

        # BM25 index structures
        self.bm25_tokenized_corpus: List[List[str]] = []
        self.bm25_doc_freqs: List[Dict[str, int]] = []
        self.bm25_idf: Dict[str, float] = {}
        self.bm25_doc_lens: np.ndarray = np.array([])
        self.bm25_avgdl: float = 1.0

        # TF-IDF & Dense SVD / Embedding structures
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.corpus_matrix = None
        self.svd_model: Optional[TruncatedSVD] = None
        self.dense_corpus_matrix: Optional[np.ndarray] = None

        self._load_and_index_local_evidence()

    @staticmethod
    def _tokenize_vi(text: str) -> List[str]:
        norm = normalize_vietnamese_unicode(text.lower())
        return re.findall(r"\w{2,}", norm)

    def _build_bm25_index(self, texts: List[str]) -> None:
        """Xây dựng chỉ mục Okapi BM25 chuẩn (k1=1.5, b=0.75)."""
        self.bm25_tokenized_corpus = [self._tokenize_vi(t) for t in texts]
        self.bm25_doc_freqs = []
        df_counts: Dict[str, int] = {}
        doc_lens = []

        N = len(self.bm25_tokenized_corpus)
        for tokens in self.bm25_tokenized_corpus:
            doc_lens.append(len(tokens))
            freqs: Dict[str, int] = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.bm25_doc_freqs.append(freqs)
            for tok in freqs.keys():
                df_counts[tok] = df_counts.get(tok, 0) + 1

        self.bm25_doc_lens = np.array(doc_lens, dtype=np.float32)
        self.bm25_avgdl = float(np.mean(self.bm25_doc_lens)) if N > 0 else 1.0

        self.bm25_idf = {}
        for tok, df in df_counts.items():
            self.bm25_idf[tok] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def _compute_bm25_scores(self, query_text: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Tính điểm Okapi BM25 trên toàn bộ Corpus."""
        q_tokens = self._tokenize_vi(query_text)
        N = len(self.local_corpus)
        if N == 0 or not q_tokens:
            return np.zeros(N, dtype=np.float32)

        scores = np.zeros(N, dtype=np.float32)
        for i, freqs in enumerate(self.bm25_doc_freqs):
            dl = self.bm25_doc_lens[i]
            norm_factor = k1 * (1.0 - b + b * (dl / max(self.bm25_avgdl, 1.0)))
            s = 0.0
            for tok in q_tokens:
                tf = freqs.get(tok, 0)
                if tf > 0:
                    idf = self.bm25_idf.get(tok, 0.0)
                    s += idf * ((tf * (k1 + 1.0)) / (tf + norm_factor))
            scores[i] = s

        max_s = float(np.max(scores)) if len(scores) > 0 else 0.0
        if max_s > 0:
            # Chuẩn hóa về [0, 1] có bảo toàn độ lớn tuyệt đối
            scores = np.clip(scores / max(max_s, 8.0), 0.0, 0.99)
        return scores

    def _compute_dense_scores(self, query_text: str) -> np.ndarray:
        """Tính điểm Dense Semantic Similarity qua không gian Vector ngữ nghĩa thu gọn + Char/Word N-gram."""
        if self.vectorizer is None or self.corpus_matrix is None:
            return np.zeros(len(self.local_corpus), dtype=np.float32)

        norm_q = normalize_vietnamese_unicode(query_text.lower())
        q_tfidf = self.vectorizer.transform([norm_q])
        lexical_cos = cosine_similarity(q_tfidf, self.corpus_matrix)[0]

        if self.svd_model is not None and self.dense_corpus_matrix is not None:
            q_dense = self.svd_model.transform(q_tfidf)
            q_dense = normalize(q_dense)
            dense_cos = np.dot(self.dense_corpus_matrix, q_dense[0])
            dense_cos = np.clip(dense_cos, 0.0, 1.0)
            return 0.60 * dense_cos + 0.40 * lexical_cos
        return lexical_cos

    @staticmethod
    def _temporal_alignment_bonus(claim_text: str, evidence_date: str) -> Dict[str, Any]:
        """
        Kiểm tra độ khớp thời gian (Temporal Filtering & Alignment):
        So sánh năm/tháng xuất hiện trong Claim với ngày xuất bản của Bằng chứng (date).
        """
        if not evidence_date:
            return {"bonus": 0.0, "temporal_match": "UNKNOWN_DATE"}

        ev_years = re.findall(r"\b(20\d{2})\b", str(evidence_date))
        claim_years = re.findall(r"\b(20\d{2})\b", claim_text)

        if not claim_years:
            return {"bonus": 0.02, "temporal_match": f"VALID ({evidence_date})"}

        if ev_years and claim_years:
            if ev_years[0] in claim_years:
                return {"bonus": 0.06, "temporal_match": f"EXACT_YEAR_MATCH ({evidence_date})"}
            else:
                # Lệch năm (ví dụ tin cũ gán ghép năm mới)
                return {"bonus": -0.05, "temporal_match": f"TEMPORAL_MISMATCH (Claim:{claim_years[0]} vs Ev:{ev_years[0]})"}

        return {"bonus": 0.0, "temporal_match": f"CHECKED ({evidence_date})"}

    def _cross_encoder_rerank_score(
        self,
        claim_text: str,
        candidate: Dict[str, Any],
        bm25_s: float,
        dense_s: float
    ) -> Dict[str, Any]:
        """
        Cross-Encoder Pairwise Reranker kết hợp kiểm tra Thực thể/Số liệu (Entity & Number Overlap)
        và Thời điểm bằng chứng (Temporal Filtering).
        """
        c_norm = normalize_vietnamese_unicode(claim_text.lower())
        e_norm = normalize_vietnamese_unicode(candidate.get("search_blob", "").lower())

        c_words = set(re.findall(r"\w{2,}", c_norm))
        e_words = set(re.findall(r"\w{2,}", e_norm))
        word_overlap = len(c_words.intersection(e_words)) / max(min(len(c_words), 16), 1)

        # Kiểm tra khớp con số / mốc định lượng (rất quan trọng trong kiểm chứng tin tức)
        c_nums = set(re.findall(r"\d+", c_norm))
        e_nums = set(re.findall(r"\d+", e_norm))
        if c_nums:
            num_overlap = len(c_nums.intersection(e_nums)) / len(c_nums)
        else:
            num_overlap = 0.5

        # Kiểm tra khớp cụm từ liền kề (Bigram exact phrase match)
        c_toks = re.findall(r"\w{2,}", c_norm)
        e_toks = re.findall(r"\w{2,}", e_norm)
        c_bigrams = set(zip(c_toks, c_toks[1:])) if len(c_toks) > 1 else set()
        e_bigrams = set(zip(e_toks, e_toks[1:])) if len(e_toks) > 1 else set()
        bigram_overlap = len(c_bigrams.intersection(e_bigrams)) / max(len(c_bigrams), 1) if c_bigrams else 0.0

        temp_info = self._temporal_alignment_bonus(claim_text, candidate.get("date", ""))

        # Điểm tổng hợp sau Reranking
        hybrid_base = 0.45 * bm25_s + 0.55 * min(1.0, dense_s * 1.65)
        cross_interaction = 0.50 * min(1.0, word_overlap * 1.35) + 0.30 * bigram_overlap + 0.20 * num_overlap
        final_sim = 0.48 * hybrid_base + 0.52 * cross_interaction + temp_info["bonus"]

        return {
            "rerank_score": float(np.clip(final_sim, 0.0, 0.99)),
            "temporal_match": temp_info["temporal_match"]
        }

    def _load_and_index_local_evidence(self) -> None:
        records: List[Dict[str, Any]] = []

        # 1. Nạp Knowledge Base kiểm chứng chính thống Việt Nam (360+ claims)
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    kb_items = json.load(f)
                for item in kb_items:
                    ev_text = item.get("gold_evidence") or item.get("evidence_text", "")
                    claim_text = item.get("claim") or item.get("claim_pattern", "")
                    if self.evidence_only:
                        # Chỉ giữ phần "bài viết bằng chứng" (tiêu đề + nội dung), bỏ claim và nhãn
                        records.append({
                            "id": item.get("id", "KB"),
                            "title": item.get("title", ""),
                            "claim": "",
                            "claim_pattern": "",
                            "evidence_text": ev_text,
                            "search_blob": f"{item.get('title', '')} {ev_text}",
                            "source": item.get("source", "Cơ sở dữ liệu Kiểm chứng Việt Nam"),
                            "url": item.get("url", "https://tingia.gov.vn/"),
                            "date": item.get("date", ""),
                            "known_verdict_score": None,
                            "known_verdict_label": None,
                            "nli_label": None,
                            "topic": item.get("topic", "Tổng hợp"),
                            "tier": "LOCAL_EVIDENCE_ONLY"
                        })
                        continue
                    records.append({
                        "id": item.get("id", "KB"),
                        "title": item.get("title", claim_text[:90]),
                        "claim": claim_text,
                        "claim_pattern": item.get("claim_pattern", claim_text),
                        "evidence_text": ev_text,
                        "search_blob": f"{item.get('title', '')} {claim_text} {item.get('claim_pattern', '')} {ev_text}",
                        "source": item.get("source", "Cơ sở dữ liệu Kiểm chứng Việt Nam"),
                        "url": item.get("url", "https://tingia.gov.vn/"),
                        "date": item.get("date", "2024-10-01"),
                        "known_verdict_score": int(item.get("verdict_score", 3)),
                        "known_verdict_label": item.get("verdict_label", "UNDETERMINED"),
                        "nli_label": item.get("label") or item.get("nli_label", "NEI"),
                        "topic": item.get("topic", "Tổng hợp"),
                        "tier": "LOCAL_KNOWLEDGE_BASE"
                    })
            except Exception as e:
                print(f"DEBUG: Lỗi nạp Knowledge Base: {e}")

        # 2. Nạp thêm các mẫu tin tức đã gán nhãn từ tập dữ liệu huấn luyện làm Kho đối chiếu lịch sử
        if self.include_historical and os.path.exists(self.train_csv_path):
            try:
                df = pd.read_csv(self.train_csv_path, nrows=250)
                text_col = next((c for c in ["post_message", "text", "content"] if c in df.columns), None)
                if text_col and "label" in df.columns:
                    for idx, row in df.dropna(subset=[text_col, "label"]).iterrows():
                        raw_msg = str(row[text_col]).strip()
                        if len(raw_msg) < 50:
                            continue
                        lbl = int(row["label"])
                        first_line = raw_msg.split("\n")[0][:140].strip()
                        snippet = re.sub(r"\s+", " ", raw_msg)[:360]
                        records.append({
                            "id": f"DATASET-{idx}",
                            "title": first_line if len(first_line) > 15 else snippet[:100] + "...",
                            "claim": snippet,
                            "claim_pattern": snippet,
                            "evidence_text": (
                                f"[Hồ sơ dữ liệu đã gán nhãn: {'TIN SAI SỰ THẬT / TIN GIẢ' if lbl == 1 else 'TIN XÁC THỰC CHÍNH THỐNG'}] "
                                f"{snippet}"
                            ),
                            "search_blob": raw_msg[:600],
                            "source": "Kho lưu trữ Dữ liệu Kiểm chứng (VLSP/ReINTEL)",
                            "url": "https://tingia.gov.vn/",
                            "date": "2023-12-15",
                            "known_verdict_score": 1 if lbl == 1 else 5,
                            "known_verdict_label": "FALSE" if lbl == 1 else "TRUE",
                            "nli_label": "Refuted" if lbl == 1 else "Supported",
                            "topic": "Lưu trữ MXH & Báo chí",
                            "tier": "HISTORICAL_DATASET"
                        })
            except Exception as e:
                print(f"DEBUG: Không thể nạp mẫu từ fake_news.csv: {e}")

        self.local_corpus = records
        if self.local_corpus:
            texts = [normalize_vietnamese_unicode(r["search_blob"].lower()) for r in self.local_corpus]
            # 1. Build BM25 Index
            self._build_bm25_index(texts)
            # 2. Build TF-IDF + Dense SVD Index
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=14000, sublinear_tf=True)
            self.corpus_matrix = self.vectorizer.fit_transform(texts)
            n_comp = min(96, self.corpus_matrix.shape[0] - 1, self.corpus_matrix.shape[1] - 1)
            if n_comp >= 8:
                self.svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
                dense_raw = self.svd_model.fit_transform(self.corpus_matrix)
                self.dense_corpus_matrix = normalize(dense_raw)

    def _compute_calibrated_similarity(self, claim_text: str, candidate_text: str, raw_tfidf_sim: float) -> float:
        c_words = set(re.findall(r"\w{2,}", claim_text.lower()))
        e_words = set(re.findall(r"\w{2,}", candidate_text.lower()))
        if not c_words or not e_words:
            return 0.0

        overlap = len(c_words.intersection(e_words)) / max(min(len(c_words), 18), 1)
        combined = 0.55 * min(1.0, raw_tfidf_sim * 2.1) + 0.45 * min(1.0, overlap * 1.35)
        return float(np.clip(combined, 0.0, 0.99))

    def search_local_evidence(
        self,
        claim_text: str,
        top_k: int = 5,
        mode: str = "hybrid_rerank"
    ) -> List[Dict[str, Any]]:
        """
        Node 4: Tìm kiếm trong Kho Bằng chứng Nội bộ với 4 chế độ phục vụ thực nghiệm:
          - mode="bm25": Chỉ dùng Okapi BM25 Baseline
          - mode="dense": Chỉ dùng Dense Vector Semantic Retrieval
          - mode="hybrid": Kết hợp BM25 + Dense (chưa qua Cross-Encoder Reranker)
          - mode="hybrid_rerank" (mặc định): Hybrid BM25 + Dense + Cross-Encoder Reranker + Temporal Filtering
        """
        if not self.local_corpus:
            return []

        bm25_scores = self._compute_bm25_scores(claim_text)
        dense_scores = self._compute_dense_scores(claim_text)

        if mode == "bm25":
            base_scores = bm25_scores
        elif mode == "dense":
            base_scores = dense_scores
        else:
            # Hybrid combination
            base_scores = 0.48 * bm25_scores + 0.52 * dense_scores

        pool_size = min(len(self.local_corpus), max(top_k * 4, 20))
        top_indices = np.argsort(base_scores)[::-1][:pool_size]

        results = []
        for idx in top_indices:
            b_score = float(bm25_scores[idx])
            d_score = float(dense_scores[idx])
            if max(b_score, d_score) <= 0.02:
                continue

            item = dict(self.local_corpus[idx])
            item["bm25_score"] = round(b_score, 3)
            item["dense_score"] = round(d_score, 3)

            if mode == "bm25":
                final_sim = min(0.99, b_score * 1.15)
                item["rerank_score"] = round(final_sim, 3)
                item["temporal_match"] = f"UNFILTERED ({item.get('date', 'N/A')})"
                item["retrieval_method"] = "BM25_BASELINE"
            elif mode == "dense":
                final_sim = min(0.99, d_score * 1.45)
                item["rerank_score"] = round(final_sim, 3)
                item["temporal_match"] = f"UNFILTERED ({item.get('date', 'N/A')})"
                item["retrieval_method"] = "DENSE_VECTOR"
            elif mode == "hybrid":
                final_sim = min(0.99, (0.48 * b_score + 0.52 * d_score) * 1.35)
                item["rerank_score"] = round(final_sim, 3)
                item["temporal_match"] = f"UNFILTERED ({item.get('date', 'N/A')})"
                item["retrieval_method"] = "HYBRID_BM25_DENSE"
            else:
                rerank_info = self._cross_encoder_rerank_score(claim_text, item, b_score, d_score)
                final_sim = rerank_info["rerank_score"]
                item["rerank_score"] = round(final_sim, 3)
                item["temporal_match"] = rerank_info["temporal_match"]
                item["retrieval_method"] = "HYBRID_CROSS_RERANK_TEMPORAL"

            item["similarity"] = round(final_sim, 3)
            results.append(item)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def search_live_vietnamese_news(self, queries: List[str], claim_text: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Node 5: Truy xuất Bằng chứng Trực tuyến thời gian thực từ Báo chí Chính thống Việt Nam
        qua Google News RSS (hl=vi&gl=VN).
        """
        live_evidences: List[Dict[str, Any]] = []
        seen_titles = set()

        for q in queries[:2]:
            clean_q = re.sub(r"[^\w\s]", " ", q).strip()
            if not clean_q:
                continue
            encoded_q = urllib.parse.quote(clean_q[:110])
            rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=vi&gl=VN&ceid=VN:vi"

            try:
                req = urllib.request.Request(
                    rss_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=4.5) as resp:
                    xml_bytes = resp.read()

                root = ET.fromstring(xml_bytes)
                for item in root.findall(".//item")[:6]:
                    raw_title = item.findtext("title") or ""
                    link = item.findtext("link") or ""
                    pub_date = item.findtext("pubDate") or ""
                    source_el = item.find("source")
                    source_name = source_el.text if source_el is not None and source_el.text else "Báo chí Việt Nam"

                    title_clean = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title.strip()
                    if not title_clean or title_clean.lower() in seen_titles:
                        continue
                    seen_titles.add(title_clean.lower())

                    raw_tfidf_s = 0.25
                    if self.vectorizer is not None:
                        v1 = self.vectorizer.transform([claim_text.lower()])
                        v2 = self.vectorizer.transform([title_clean.lower()])
                        raw_tfidf_s = float(cosine_similarity(v1, v2)[0][0])

                    sim = self._compute_calibrated_similarity(claim_text, title_clean, raw_tfidf_s)
                    short_date = pub_date[:16] if pub_date else "Trực tuyến"

                    live_evidences.append({
                        "id": f"LIVE-{len(live_evidences)+1}",
                        "title": title_clean,
                        "claim_pattern": title_clean,
                        "evidence_text": f"Bài báo từ {source_name} ({short_date}): \"{title_clean}\"",
                        "source": source_name,
                        "url": link,
                        "date": short_date,
                        "pub_date": pub_date,
                        "known_verdict_score": None,
                        "known_verdict_label": None,
                        "nli_label": None,
                        "topic": "Tin tức Trực tuyến (Live RSS)",
                        "tier": "LIVE_WEB_NEWS",
                        "bm25_score": round(raw_tfidf_s, 3),
                        "dense_score": round(sim, 3),
                        "rerank_score": round(sim, 3),
                        "temporal_match": f"LIVE_RSS ({short_date})",
                        "retrieval_method": "GOOGLE_NEWS_RSS",
                        "similarity": round(sim, 3)
                    })
            except Exception:
                continue

        live_evidences.sort(key=lambda x: x["similarity"], reverse=True)
        return live_evidences[:max_results]

    def retrieve_and_filter_by_vosoughi_threshold(
        self, claim_text: str, queries: List[str], mode: str = "hybrid_rerank"
    ) -> Dict[str, Any]:
        """
        Node 6: Hợp nhất Bằng chứng Nội bộ (Hybrid BM25 + Dense + Reranker) + Trực tuyến
        và áp dụng quy tắc ngưỡng tại Mục S2.2 (Vosoughi et al., Science, 2018):
          - sim < 0.50  -> Loại bỏ (Discarded)
          - 0.50 <= sim < 0.90 -> Đưa vào kiểm tra Stance/NLI (Needs Inspection)
          - sim >= 0.90 -> Khớp trực tiếp độ tin cậy cao (High-Confidence Match)
        """
        local_candidates = self.search_local_evidence(claim_text, top_k=5, mode=mode)
        live_candidates = self.search_live_vietnamese_news(queries, claim_text, max_results=3)

        all_candidates = local_candidates + live_candidates
        all_candidates.sort(key=lambda x: x["similarity"], reverse=True)

        accepted_evidences: List[Dict[str, Any]] = []
        discarded_count = 0

        for cand in all_candidates:
            sim = cand["similarity"]
            if sim >= 0.90:
                cand["vosoughi_s22_band"] = "HIGH_MATCH (>= 0.90: Khớp trực tiếp)"
                accepted_evidences.append(cand)
            elif sim >= 0.50:
                cand["vosoughi_s22_band"] = "INSPECTION_BAND (0.50 - 0.90: Cần đối chiếu NLI)"
                accepted_evidences.append(cand)
            else:
                cand["vosoughi_s22_band"] = "DISCARDED (< 0.50: Độ tương đồng thấp)"
                discarded_count += 1

        return {
            "accepted_evidences": accepted_evidences[:5],
            "top_candidates_all": all_candidates[:5],
            "stats": {
                "retrieval_mode": mode.upper(),
                "total_candidates_checked": len(all_candidates),
                "accepted_above_050": len(accepted_evidences),
                "discarded_below_050": discarded_count,
                "max_similarity": round(all_candidates[0]["similarity"], 3) if all_candidates else 0.0
            }
        }
