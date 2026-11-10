import pandas as pd
import torch
import os
import sys
import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import Dataset
from tqdm import tqdm

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }

def print_header(step, title):
    print(f"\n" + "="*80)
    print(f" [{step}/7] {title.upper()}")
    print("="*80)
    time.sleep(1.2)

def load_data_from_dir(dir_path):
    if not os.path.exists(dir_path):
        return pd.DataFrame()
    all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.csv')]
    if not all_files:
        return pd.DataFrame()
    
    dfs = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            text_col = next((c for c in ['post_message', 'text', 'content', 'Maintext', 'maintext'] if c in temp_df.columns), None)
            label_col = next((c for c in temp_df.columns if c.lower() == 'label'), None)
            
            if text_col and label_col:
                temp_df = temp_df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
                dfs.append(temp_df)
                print(f"    [+] Loaded: {os.path.basename(f)} ({len(temp_df)} records)")
        except:
            pass
            
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def train_phobert(data_dir, models_dir, epochs=10):
    start_total = time.time()
    
    # --- BUOC 1: DOC DU LIEU ---
    print_header(1, "Khoi tao va Doc du lieu nguon")
    df_train = load_data_from_dir(os.path.join(data_dir, 'train'))
    df_val = load_data_from_dir(os.path.join(data_dir, 'val'))
    df_test = load_data_from_dir(os.path.join(data_dir, 'test'))

    if df_train.empty or df_test.empty or df_val.empty:
        print(" [ERROR] Khong tim thay du lieu hop le trong cac thu muc train, val hoac test.")
        return
    
    print(f"  - Tong so mau: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # --- BUOC 2: PREPROCESSING ---
    print_header(2, "Lam sach van ban cho Transformer")
    print("  - [1/2] Dang thuc hien lam sach van ban tieng Viet...")
    df_train['clean_text'] = df_train['text'].apply(clean_text)
    df_val['clean_text'] = df_val['text'].apply(clean_text)
    df_test['clean_text'] = df_test['text'].apply(clean_text)
    
    X_train, y_train = df_train['clean_text'].tolist(), df_train['label'].tolist()
    X_val, y_val = df_val['clean_text'].tolist(), df_val['label'].tolist()
    X_test, y_test = df_test['clean_text'].tolist(), df_test['label'].tolist()
    print("  - [2/2] Hoan tat chuan hoa van ban.")

    # --- BUOC 3: TOKENIZATION ---
    print_header(3, "Ma hoa du lieu (Tokenization)")
    model_name = "vinai/phobert-base-v2"
    print(f"  - [1/3] Dang tai PhoBERT Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"  - [2/3] Dang chuyen doi van ban thanh chuoi Token ID (Max length: 128)...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)
    
    print(f"  - [3/3] Dang khoi tao PyTorch Datasets...")
    train_dataset = FakeNewsDataset(train_encodings, y_train)
    val_dataset = FakeNewsDataset(val_encodings, y_val)
    test_dataset = FakeNewsDataset(test_encodings, y_test)

    # --- BUOC 4: KHOI TAO MO HINH PHOBERT ---
    print_header(4, "Thiet lap kien truc Transformer")
    print(f"  - [1/2] Dang tai phan trong so Pre-trained tu VinAI...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    print(f"  - [2/2] Dang cau hinh tham so huan luyen (Learning Rate, Batch Size)...")
    output_dir = os.path.join(models_dir, "phobert_checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
        report_to="none",
    )

    # --- BUOC 5: FINE-TUNING MO HINH ---
    print_header(5, "Bat dau Fine-tuning PhoBERT")
    print("  [GIAI THICH]: AI dang hoc cach 'chu y' (Attention) vao cac tu ngu quan trong...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("  - Tien trinh huan luyen (co the mat nhieu thoi gian nhat):")
    trainer.train()

    # --- BUOC 6: DANH GIA VA LUU TRU ---
    print_header(6, "Danh gia cuoi cung va Luu mo hinh")
    print("  - [1/3] Dang tinh toan metrics tren tap du lieu Test (an)...")
    eval_results = trainer.evaluate(test_dataset)
    
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"  - [2/3] Chi so: Acc={eval_results['eval_accuracy']*100:.1f}% | F1={eval_results['eval_f1']*100:.1f}%")
    print(f"  - [3/3] Dang luu model va tokenizer tai: models/phobert_model")
    
    phobert_save_path = os.path.join(models_dir, "phobert_model")
    model.save_pretrained(phobert_save_path)
    tokenizer.save_pretrained(phobert_save_path)

    # Cap nhat metrics.json
    metrics_path = os.path.join(models_dir, "models_metrics.json")
    models_info = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f: models_info = json.load(f)
        except: pass

    duration = time.time() - start_total
    models_info["phobert"] = {
        "accuracy": float(eval_results['eval_accuracy']),
        "f1": float(eval_results['eval_f1']),
        "precision": float(eval_results['eval_precision']),
        "recall": float(eval_results['eval_recall']),
        "false_alarm_rate": float(far),
        "duration": float(duration),
        "confusion_matrix": cm.tolist()
    }
    with open(metrics_path, "w") as f:
        json.dump(models_info, f, indent=4)

    # --- BUOC 7: TONG KET ---
    print_header(7, "Hoan tat quy trinh PhoBERT")
    print(f"  - Tong thoi gian huan luyen Deep Learning: {duration:.2f} giay")
    print("="*80 + "\n")

if __name__ == "__main__":
    DATA_DIR = "data"
    MODELS_DIR = "models"
    train_phobert(DATA_DIR, MODELS_DIR)
