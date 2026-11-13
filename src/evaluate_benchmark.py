import os
import sys
import time
import json
import math
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.build_vn_factcheck_dataset import build_evaluation_dataset
from src.preprocess import predict_check_worthiness
from src.evidence_retriever import MultiTierEvidenceRetriever
from src.evidence_analyzer import analyze_single_evidence_stance, run_llm_few_shot_verification


from src.evaluate_benchmark_strict import evaluate_strict


def evaluate_all_benchmarks() -> Dict[str, Any]:
    """
    Chạy bộ Benchmark Đánh giá Chặt (Strict Leakage-Free + Live Gemini 3-Shot + Ablation + Error Analysis)
    và lưu đồng bộ vào models/factcheck_benchmark_metrics.json.
    """
    return evaluate_strict(live_gemini_sample_size=12)


if __name__ == "__main__":
    evaluate_all_benchmarks()
