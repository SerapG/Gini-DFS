"""
experiment_dl.py
────────────────
Derin Öğrenme Modelleri (TextCNN, LSTM) için Gini/DFS kelime dağarcığı
(vocabulary) kısıtlamalı deneysel analiz.
Makine öğrenmesi boru hattı ile birebir aynı sonuç tablosunu üretir.
"""

import os
import sys
import warnings
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from data_loader import load_english, load_turkish
from main import build_score_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch bulunamadı! Lütfen yükleyin: pip install torch")

RESULTS_DIR = "results"
TEST_SIZE = 0.30
RANDOM_STATE = 42
MAX_LEN = 64
EPOCHS = 2
BATCH_SIZE = 16
VOCAB_SIZES = [500, 300, 100, 50, 30, 10]

SCENARIOS = [
    ("Sadece Temel", False, False),
    ("Sadece Stopword", True, False),
    ("Sadece Stemming", False, True),
    ("Hepsi (Stopword+Stem)", True, True)
]

if TORCH_AVAILABLE:
    class TextCNNModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 1)
            
        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.relu(x)
            x = torch.max(x, dim=2)[0]
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    class TextLSTMModel(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)
            
        def forward(self, x):
            x = self.embedding(x)
            _, (hn, _) = self.lstm(x)
            x = hn[-1]
            x = self.dropout(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

def _train_and_evaluate(model, X_train, y_train, X_test, y_test):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        out = model(X_test)
        preds = (out > 0).int().squeeze().numpy()
        trues = y_test.squeeze().numpy()
        
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average="weighted", zero_division=0)
    rec = recall_score(trues, preds, average="weighted", zero_division=0)
    f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    
    return round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)

def run_dl_pipeline(df: pd.DataFrame, scenario_name: str, lang_label: str) -> list:
    if not TORCH_AVAILABLE:
        return []
        
    print(f"\n{'─' * 62}")
    print(f"  [{lang_label} SMS] okunuyor... (Senaryo: {scenario_name})")
    print(f"  → Toplam mesaj      : {len(df)}")
    
    matrix = build_score_matrix(df, f"{lang_label} SMS")
    
    labels = df["label"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    texts = df["tokens"].tolist()
    
    X_train_raw, X_test_raw, y_tr, y_te = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    y_train = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)
    
    all_rows = []
    
    for vocab_size in VOCAB_SIZES:
        gini_vocab = matrix.head(vocab_size)["Kelime"].tolist()
        dfs_vocab = matrix.sort_values("DFS_Score", ascending=False).head(vocab_size)["Kelime"].tolist()
        
        for method_name, vocab in [("Gini", gini_vocab), ("DFS", dfs_vocab)]:
            word2idx = {w: i+2 for i, w in enumerate(vocab)}
            word2idx["<PAD>"] = 0
            word2idx["<OOV>"] = 1
            vocab_dim = len(vocab) + 2
            
            def vectorize(tokens_list):
                seqs = []
                for toks in tokens_list:
                    seq = [word2idx.get(w, 1) for w in toks][:MAX_LEN]
                    seq = seq + [0] * max(0, MAX_LEN - len(seq))
                    seqs.append(seq)
                return torch.tensor(seqs, dtype=torch.long)
                
            X_train = vectorize(X_train_raw)
            X_test = vectorize(X_test_raw)
            
            # TextCNN
            cnn_model = TextCNNModel(vocab_dim, 32)
            acc, prec, rec, f1 = _train_and_evaluate(cnn_model, X_train, y_train, X_test, y_test)
            all_rows.append({
                "On_Isleme": scenario_name,
                "Dil": lang_label,
                "Kelime_Sayisi": vocab_size,
                "Yontem": method_name,
                "Algoritma": "TextCNN",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
            })
            print(f"      ✔ TextCNN        | {method_name:<4} | Vocab={vocab_size:<3} | F1={f1:.4f}")
            
            # LSTM
            lstm_model = TextLSTMModel(vocab_dim, 32)
            acc, prec, rec, f1 = _train_and_evaluate(lstm_model, X_train, y_train, X_test, y_test)
            all_rows.append({
                "On_Isleme": scenario_name,
                "Dil": lang_label,
                "Kelime_Sayisi": vocab_size,
                "Yontem": method_name,
                "Algoritma": "LSTM",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
            })
            print(f"      ✔ LSTM           | {method_name:<4} | Vocab={vocab_size:<3} | F1={f1:.4f}")
            
    return all_rows

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    
    print("\n" + "="*65)
    print(" 🧠 DERİN ÖĞRENME MODEL EĞİTİMİ (Gini/DFS Sabit Vocab)")
    print("="*65)
    
    for sc_name, rm_stop, ap_stem in SCENARIOS:
        # Ingilizce
        df_en = load_english(remove_stopwords=rm_stop, apply_stemming=ap_stem)
        all_rows += run_dl_pipeline(df_en, sc_name, "Ingilizce")
        
        # Turkce
        df_tr = load_turkish(remove_stopwords=rm_stop, apply_stemming=ap_stem)
        all_rows += run_dl_pipeline(df_tr, sc_name, "Turkce")
        
    final_df = pd.DataFrame(all_rows, columns=[
        "On_Isleme", "Dil", "Kelime_Sayisi", "Yontem", "Algoritma",
        "Accuracy", "Precision", "Recall", "F1_Score"
    ])
    
    out_path = os.path.join(RESULTS_DIR, "dl_comparison_results.csv")
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    print(f"\n✓ Derin Öğrenme Pipeline tamamlandı! Sonuçlar: {out_path}")

if __name__ == "__main__":
    main()