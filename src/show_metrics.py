import json
import os

metrics_path = os.path.join("models", "models_metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            for k, v in m.items():
                if isinstance(v, dict) and "accuracy" in v:
                    acc = v.get("accuracy", 0)
                    f1 = v.get("f1", 0)
                    print(f"- {k.upper()}: Accuracy={acc:.4f}, F1={f1:.4f}")
    except Exception as e:
        print(f"[!] Error reading metrics: {e}")
else:
    print("[!] Metrics file not found.")
