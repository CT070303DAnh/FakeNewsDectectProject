import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(input_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Splits the CSV data into train, validation, and test sets.
    """
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Clean data (basic dropna for essential columns)
    text_col = 'post_message' if 'post_message' in df.columns else 'text'
    df.dropna(subset=[text_col, 'label'], inplace=True)
    
    print(f"Total samples after cleaning: {len(df)}")
    
    # 2. Split: Train vs (Val + Test)
    df_train, df_temp = train_test_split(
        df, test_size=(1 - train_ratio), random_state=random_state, stratify=df['label']
    )
    
    # 3. Split: Val vs Test
    # relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp, test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=random_state, stratify=df_temp['label']
    )
    
    # 4. Save files into separate folders
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_path = os.path.join(train_dir, 'train.csv')
    val_path = os.path.join(val_dir, 'val.csv')
    test_path = os.path.join(test_dir, 'test.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Splitting complete!")
    print(f" - Train folder: {train_dir} (contains train.csv)")
    print(f" - Val folder:   {val_dir} (contains val.csv)")
    print(f" - Test folder:  {test_dir} (contains test.csv)")

if __name__ == "__main__":
    INPUT_FILE = os.path.join("data", "fake_news.csv")
    OUTPUT_DIR = "data"
    split_data(INPUT_FILE, OUTPUT_DIR)
