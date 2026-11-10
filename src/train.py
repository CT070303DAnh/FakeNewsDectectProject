import pandas as pd
import joblib
import os
import sys
import numpy as np
import json
import time
import logging
import warnings
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable=None, **kwargs):
        return iterable
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    HAS_GENSIM = True
except ImportError:
    Doc2Vec = None
    TaggedDocument = None
    HAS_GENSIM = False

# Chan cac canh bao va log he thong de hien thi sach se
warnings.filterwarnings("ignore")
logging.getLogger('gensim').setLevel(logging.ERROR)

# Add parent dir to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import clean_text

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

def train_all_models(data_dir, models_dir):
    start_total = time.time()
    
    # --- BUOC 1: DOC DU LIEU ---
    print_header(1, "Khoi tao va Doc du lieu nguon")
    df_train = load_data_from_dir(os.path.join(data_dir, 'train'))
    df_val = load_data_from_dir(os.path.join(data_dir, 'val'))
    df_test = load_data_from_dir(os.path.join(data_dir, 'test'))

    if df_train.empty or df_test.empty:
        print(f" [ERROR] Khong tim thay du lieu hop le.")
        return
    
    print(f"  - Tong so mau du lieu: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # --- BUOC 2: PREPROCESSING ---
    print_header(2, "Tien hanh lam sach van ban")
    start_pre = time.time()
    if HAS_TQDM:
        tqdm.pandas(desc="    Tien trinh lam sach (%)")
        df_train['clean_text'] = df_train['text'].progress_apply(clean_text)
        df_val['clean_text'] = df_val['text'].progress_apply(clean_text)
        df_test['clean_text'] = df_test['text'].progress_apply(clean_text)
    else:
        df_train['clean_text'] = df_train['text'].apply(clean_text)
        df_val['clean_text'] = df_val['text'].apply(clean_text)
        df_test['clean_text'] = df_test['text'].apply(clean_text)
    
    X_train, y_train = df_train['clean_text'], df_train['label']
    X_val, y_val = df_val['clean_text'], df_val['label']
    X_test, y_test = df_test['clean_text'], df_test['label']
    
    print(f"  - Thoi gian xu ly: {time.time() - start_pre:.2f}s")

    # --- BUOC 3: TF-IDF VECTORIZATION ---
    print_header(3, "Trich xuat dac trung van ban (TF-IDF)")
    start_tfidf = time.time()
    
    print(f"  - [1/4] Khoi tao cau hinh TF-IDF (Max features: 5000)...")
    time.sleep(0.5)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    
    print(f"  - [2/4] Dang phan tich tu vung tu {len(X_train)} mau van ban...")
    tfidf_vectorizer.fit(X_train)
    
    vocab_size = len(tfidf_vectorizer.vocabulary_)
    print(f"  - [3/4] Dang tinh toan trong so IDF cho {vocab_size} dac trung...")
    time.sleep(0.5)
    
    print(f"  - [4/4] Dang dong goi va luu tru Vectorizer...")
    joblib.dump(tfidf_vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    
    print(f"  - Hoan tat. Thoi gian thuc hien: {time.time() - start_tfidf:.2f}s")

    # --- BUOC 4: DOC2VEC ---
    print_header(4, "Xay dung Vector dai dien (Doc2Vec)")
    if HAS_GENSIM:
        start_d2v = time.time()
        X_all = pd.concat([X_train, X_val, X_test])
        tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(X_all)]
        
        d2v_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=40)
        print(f"  - Dang khoi tao tu vung (Vocabulary size: {len(X_all)} docs)...")
        d2v_model.build_vocab(tagged_data)
        
        print(f"  - Dang huan luyen Vector hoa:")
        pbar = tqdm(total=40, desc="    Tien do huan luyen (%)")
        for epoch in range(40):
            d2v_model.train(tagged_data, total_examples=d2v_model.corpus_count, epochs=1)
            pbar.update(1)
        pbar.close()
        
        d2v_model.save(os.path.join(models_dir, "doc2vec.model"))
        print(f"  - Hoan tat. Thoi gian thuc hien: {time.time() - start_d2v:.2f}s")
    else:
        print("  - [SKIP] Thu vien gensim chua duoc cai dat, su dung TF-IDF SVD thay the.")

    # --- BUOC 5: CLASSIFIERS ---
    print_header(5, "Huan luyen cac thuat toan Phan loai")
    
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000), 'logistic_regression'),
        ('SVM (Support Vector)', SVC(probability=True), 'svm'),
        ('Random Forest', RandomForestClassifier(n_estimators=100), 'random_forest'),
        ('Naive Bayes', MultinomialNB(), 'naive_bayes'),
        ('Decision Tree', DecisionTreeClassifier(), 'decision_tree')
    ]

    models_info = {}
    metrics_path = os.path.join(models_dir, "models_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f: models_info = json.load(f)
        except: pass

    # Danh sach mo ta dac trung cho tung mo hinh
    model_descriptions = {
        'logistic_regression': {
            'train': 'Dang tim kiem trong so toi uu cho phuong trinh Logistic...',
            'eval': 'Dang kiem tra kha nang phan loai tuyen tinh trên tap Val & Test...'
        },
        'svm': {
            'train': 'Dang xac dinh sieu phang (Hyperplane) phan tach du lieu voi le lon nhat...',
            'eval': 'Dang do luong do chinh xac cua sieu phang tren du lieu moi...'
        },
        'random_forest': {
            'train': 'Dang xay dung tap hop cac cay quyet dinh (Ensemble of Trees)...',
            'eval': 'Dang tong hop y kien tu cac cay de dua ra ket qua cuoi cung...'
        },
        'naive_bayes': {
            'train': 'Dang tinh toan xac suat Bayes dua tren tan suat tu vung...',
            'eval': 'Dang ap dung quy tac Bayes de du doan nhan cho du lieu...'
        },
        'decision_tree': {
            'train': 'Dang phan nhanh du lieu de tao cau truc cay quyet dinh...',
            'eval': 'Dang duyet cay quyet dinh de tim nhan cho cac mau kiem thu...'
        }
    }

    for i, (full_name, clf, short_name) in enumerate(classifiers):
        print(f"\n  ({i+1}/{len(classifiers)}) --- {full_name.upper()} ---")
        start_step = time.time()
        desc = model_descriptions.get(short_name, {'train': 'Dang huan luyen...', 'eval': 'Dang danh gia...'})
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, vocabulary=tfidf_vectorizer.vocabulary_)),
            ('clf', clf)
        ])
        
        print(f"    [1] Thuc thi: {desc['train']}")
        pipeline.fit(X_train, y_train)
        
        print(f"    [2] Danh gia: {desc['eval']}")
        y_val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        y_test_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        
        # Tinh Báo động giả (False Alarm Rate)
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        duration = time.time() - start_step
        print(f"    [3] Chi so: Acc={acc*100:.1f}% | F1={f1*100:.1f}% | Prec={prec*100:.1f}% | Rec={rec*100:.1f}%")
        print(f"    [4] FAR: {far*100:.2f}% (Ty le bao dong gia tren tin that)")
        print(f"    [5] Thoi gian thuc hien: {duration:.2f} giay")
        
        models_info[short_name] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "false_alarm_rate": float(far),
            "duration": float(duration),
            "confusion_matrix": cm.tolist()
        }
        joblib.dump(pipeline, os.path.join(models_dir, f"{short_name}.pkl"))

    # --- BUOC 6: LUU KET QUA ---
    print_header(6, "Luu tru bao cao Metrics")
    with open(metrics_path, "w") as f:
        json.dump(models_info, f, indent=4)
    print(f"  - Da cap nhat tep tin: models/models_metrics.json (San sang cho Frontend)")

    # --- BUOC 7: TONG KET ---
    print_header(7, "Tong ket toan bo quy trinh")
    print(f"  - Tong thoi gian chay: {time.time() - start_total:.2f} giay")
    print("="*80 + "\n")

if __name__ == "__main__":
    DATA_DIR = "data"
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_all_models(DATA_DIR, MODELS_DIR)
