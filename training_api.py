import os
import sys
import json
import time
import psutil
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
import multiprocessing
import threading
import re
import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import training functions
from src.train import train_all_models
try:
    from src.train_phobert import train_phobert
    from src.train_sbert import train_sbert
    from src.train_vibert import train_vibert
except ImportError:
    train_phobert = None
    train_sbert = None
    train_vibert = None

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for training
training_status = {
    "is_training": False,
    "current_model": None,
    "progress": 0,
    "start_time": None,
    "logs": [],
    "details": {
        "current": 0,
        "total": 0,
        "elapsed": "00:00",
        "remaining": "00:00",
        "speed": "0it/s"
    }
}

# Reference to active process
training_process = None

# Queue for inter-process communication
log_queue = multiprocessing.Queue()

class TrainingRequest(BaseModel):
    model_type: str  # 'all', 'phobert', 'sbert', 'vibert'
    epochs: Optional[int] = 10

class CustomStream:
    def __init__(self, queue, original_stream):
        self.queue = queue
        self.original_stream = original_stream

    def write(self, data):
        if data.strip():
            msg = json.dumps({"type": "log", "content": data.strip()})
            self.queue.put(msg)
        self.original_stream.write(data)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return True # Pretend to be a TTY to get progress bars

    def fileno(self):
        try:
            return self.original_stream.fileno()
        except:
            return 1

    @property
    def encoding(self):
        return getattr(self.original_stream, 'encoding', 'utf-8')

def training_process_wrapper(model_type, epochs, data_dir, models_dir, queue):
    # Redirect stdout and stderr to capture logs and progress bars
    stream = CustomStream(queue, sys.stdout)
    sys.stdout = stream
    sys.stderr = stream
    
    try:
        if model_type == 'all':
            train_all_models(data_dir, models_dir)
        elif model_type == 'phobert':
            train_phobert(data_dir, models_dir, epochs=epochs)
        elif model_type == 'sbert':
            train_sbert(data_dir, models_dir, epochs=epochs)
        elif model_type == 'vibert':
            train_vibert(data_dir, models_dir, epochs=epochs)
        
        queue.put(json.dumps({"type": "status", "status": "completed"}))
    except Exception as e:
        queue.put(json.dumps({"type": "status", "status": "failed", "error": str(e)}))

# Background thread to pull from queue and update global status
def update_global_status_worker():
    global training_status
    while True:
        try:
            data = log_queue.get()
            parsed = json.loads(data)
            
            if parsed["type"] == "log":
                content = parsed["content"]
                training_status["logs"].append(content)
                
                # Robust progress detection (tqdm style: 112/7380 [07:46<7:22:35, 3.65s/it])
                tqdm_match = re.search(r'(\d+)/(\d+)\s+\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?),\s+([^\]]+)\]', content)
                if tqdm_match:
                    curr = int(tqdm_match.group(1))
                    total = int(tqdm_match.group(2))
                    training_status["details"]["current"] = curr
                    training_status["details"]["total"] = total
                    training_status["details"]["elapsed"] = tqdm_match.group(3)
                    training_status["details"]["remaining"] = tqdm_match.group(4)
                    training_status["details"]["speed"] = tqdm_match.group(5)
                    training_status["progress"] = round((curr / total) * 100)
                
                # Fallback Epoch detection
                elif 'Epoch' in content:
                    epoch_match = re.search(r'Epoch\s+(\d+)\/(\d+)', content)
                    if epoch_match:
                        training_status["progress"] = round((int(epoch_match.group(1)) / int(epoch_match.group(2))) * 100)

            elif parsed["type"] == "status":
                if parsed["status"] in ["completed", "failed"]:
                    training_status["is_training"] = False
                    if parsed["status"] == "completed":
                        training_status["progress"] = 100
        except Exception as e:
            # Avoid crashing the worker
            pass

# Start the worker thread
threading.Thread(target=update_global_status_worker, daemon=True).start()

@app.post("/start-training")
async def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    global training_status, training_process
    if training_status["is_training"]:
        return {"error": "Training already in progress"}

    training_status["is_training"] = True
    training_status["current_model"] = req.model_type
    training_status["start_time"] = time.time()
    training_status["logs"] = []
    training_status["progress"] = 0
    training_status["details"] = {"current": 0, "total": 0, "elapsed": "00:00", "remaining": "00:00", "speed": "0it/s"}
    
    data_dir = os.path.join(BASE_DIR, "data")
    models_dir = os.path.join(BASE_DIR, "models")
    
    training_process = multiprocessing.Process(
        target=training_process_wrapper, 
        args=(req.model_type, req.epochs, data_dir, models_dir, log_queue)
    )
    training_process.start()
    
    return {"message": f"Started training for {req.model_type}"}

@app.post("/stop-training")
async def stop_training():
    global training_status, training_process
    if training_process and training_process.is_alive():
        training_process.terminate()
        training_process.join()
        training_status["is_training"] = False
        training_status["logs"].append("[SYSTEM] Huấn luyện đã bị DỪNG bởi người dùng.")
        return {"message": "Training stopped"}
    return {"error": "No training in progress"}

@app.get("/training-status")
async def get_training_status():
    return training_status

async def event_generator():
    last_idx = 0
    while True:
        if last_idx < len(training_status["logs"]):
            for i in range(last_idx, len(training_status["logs"])):
                yield f"data: {json.dumps({'type': 'log', 'content': training_status['logs'][i], 'details': training_status['details'], 'progress': training_status['progress']})}\n\n"
            last_idx = len(training_status["logs"])
        
        if not training_status["is_training"] and last_idx >= len(training_status["logs"]):
            yield f"data: {json.dumps({'type': 'status', 'status': 'completed' if training_status['progress'] == 100 else 'idle'})}\n\n"
            break
        
        await asyncio.sleep(0.5)

@app.get("/training-events")
async def training_events():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/system-stats")
async def get_system_stats():
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    
    gpu_stats = []
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_stats.append({
                "name": props.name,
                "usage": 0, # psutil doesn't give GPU usage easily, we'd need pynvml
                "memory_total": props.total_memory / (1024**2),
                "memory_used": torch.cuda.memory_allocated(i) / (1024**2)
            })
    
    return {
        "cpu": cpu_usage,
        "ram_total": ram.total / (1024**3),
        "ram_used": ram.used / (1024**3),
        "ram_percent": ram.percent,
        "gpus": gpu_stats,
        "disk": psutil.disk_usage('/').percent
    }

@app.get("/training-results")
async def get_training_results():
    metrics_path = os.path.join(BASE_DIR, "models", "models_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.get("/benchmark-metrics")
async def get_benchmark_metrics():
    bench_path = os.path.join(BASE_DIR, "models", "benchmark_metrics.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
