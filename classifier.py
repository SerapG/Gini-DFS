"""
classifier.py
─────────────
TF-IDF + Makine Ögrenmesi Siniflandirma Modülü.

Disa açilan fonksiyonlar:
  prepare_data(df, lang_label, run_no)
      → Veriyi %70/%30 böler, train.csv / test.csv kaydeder.
        (Her dil için sadece 1 kez çagrilir.)

  run_vocab_experiment(train_path, test_path, score_matrix, top_vocab, lang_label)
      → Belirli bir vocabulary boyutu için TF-IDF + SVM/MNB/RF çalistirir.
        Her çagri Dil, Kelime_Sayisi, Yontem, Algoritma, Accuracy, Precision,
        Recall, F1_Score sütunlarini içeren 6 satirlik bir liste döndürür.
"""

import os
import warnings
import pandas as pd

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm                     import SVC
from sklearn.naive_bayes             import MultinomialNB
from sklearn.ensemble                import RandomForestClassifier
from sklearn.metrics                 import (accuracy_score, precision_score,
                                             recall_score, f1_score)

warnings.filterwarnings("ignore")

# ─── SABİTLER ─────────────────────────────────────────────────────────────────

RESULTS_DIR  = "results"
TEST_SIZE    = 0.30
RANDOM_STATE = 42


# ─── YARDIMCI: Token listesini metne çevir ────────────────────────────────────

def _tokens_to_text(df: pd.DataFrame) -> list:
    """Token listesini tek string'e çevirir (TfidfVectorizer için)."""
    return [" ".join(tokens) for tokens in df["tokens"]]


# ─── 1. ADIM: Veri Bölme ve CSV Kayıt ────────────────────────────────────────

def prepare_data(
    df:         pd.DataFrame,
    lang_label: str,
    run_no:     int,
) -> tuple:
    """
    Veriyi %70 Egitim / %30 Test olarak böler ve ham metni CSV olarak kaydeder.
    TF-IDF uygulanmadan önce çagrilir; her dil için yalnizca 1 kez çagrilmali.

    Parametreler
    ------------
    df         : 'label' ve 'tokens' sütunlari olan DataFrame.
    lang_label : 'english' / 'turkish' gibi dil slug'i.
    run_no     : Bu çalistirmaya ait sira numarasi.

    Döndürür
    --------
    (train_path, test_path) : Kaydedilen CSV dosyalarinin yollari.
    """
    lang_slug = lang_label.lower().replace(" ", "_")

    texts  = _tokens_to_text(df)
    labels = df["label"].tolist()

    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = labels,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_path = os.path.join(RESULTS_DIR, f"{run_no}-{lang_slug}_train.csv")
    test_path  = os.path.join(RESULTS_DIR, f"{run_no}-{lang_slug}_test.csv")

    pd.DataFrame({"label": y_train, "text": texts_train}).to_csv(
        train_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"label": y_test, "text": texts_test}).to_csv(
        test_path, index=False, encoding="utf-8-sig"
    )

    print(f"  ✓ Train CSV kaydedildi : {train_path}  ({len(y_train)} satir)")
    print(f"  ✓ Test  CSV kaydedildi : {test_path}  ({len(y_test)} satir)")
    return train_path, test_path


# ─── 2. ADIM: TF-IDF ve Sınıflandırma (tek vocab_size için) ──────────────────

def _build_tfidf(train_path: str, test_path: str, vocabulary: list):
    """Kaydedilen CSV'lerden TF-IDF matrisi olusturur."""
    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test  = pd.read_csv(test_path,  encoding="utf-8-sig")

    texts_train = df_train["text"].fillna("").tolist()
    texts_test  = df_test["text"].fillna("").tolist()
    y_train     = df_train["label"].tolist()
    y_test      = df_test["label"].tolist()

    vec     = TfidfVectorizer(vocabulary=vocabulary, sublinear_tf=True)
    X_train = vec.fit_transform(texts_train)
    X_test  = vec.transform(texts_test)
    return X_train, X_test, y_train, y_test


def run_vocab_experiment(
    train_path:   str,
    test_path:    str,
    score_matrix: pd.DataFrame,
    top_vocab:    int,
    lang_label:   str,
) -> list:
    """
    Belirli bir vocabulary boyutu (top_vocab) için TF-IDF + SVM/MNB/RF çalistirir.

    Vocabulary, score_matrix'teki Gini ve DFS siralamalarinin ilk top_vocab
    elemanlarindan türetilir. Her iki siralama için ayri deney yapilir.

    Parametreler
    ------------
    train_path   : Egitim CSV dosyasi (prepare_data çikti).
    test_path    : Test CSV dosyasi (prepare_data çikti).
    score_matrix : Gini_Score ve DFS_Score sütunlari olan DataFrame.
    top_vocab    : Kullanilacak kelime sayisi (ör. 500, 100, 30…).
    lang_label   : Sonuç tablosundaki 'Dil' sütunu için etiket.

    Döndürür
    --------
    list[dict]  Dil, Kelime_Sayisi, Yontem, Algoritma, Accuracy,
                Precision, Recall, F1_Score sütunlarini içeren 6 satir
                (3 algoritma × 2 vocabulary yöntemi).
    """
    # ── Vocabulary hazirla ────────────────────────────────────────────────────
    gini_vocab = score_matrix.head(top_vocab)["Kelime"].tolist()
    dfs_vocab  = (
        score_matrix
        .sort_values("DFS_Score", ascending=False)
        .head(top_vocab)["Kelime"]
        .tolist()
    )

    classifiers = {
        "SVM":           SVC(kernel="linear", random_state=RANDOM_STATE),
        "MNB":           MultinomialNB(),
        "Random Forest": RandomForestClassifier(
                             n_estimators=100,
                             random_state=RANDOM_STATE,
                             n_jobs=-1,
                         ),
    }

    results = []

    for vocab_method, vocabulary in [("Gini", gini_vocab), ("DFS", dfs_vocab)]:
        # ── CSV'den oku → TF-IDF ──────────────────────────────────────────────
        X_train, X_test, y_train, y_test = _build_tfidf(
            train_path, test_path, vocabulary
        )

        for name, clf in classifiers.items():
            X_tr = X_train.toarray() if name == "Random Forest" else X_train
            X_te = X_test.toarray()  if name == "Random Forest" else X_test

            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)

            acc = round(accuracy_score(y_test, y_pred), 4)
            f1  = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)

            results.append({
                "Dil":           lang_label,
                "Kelime_Sayisi": top_vocab,
                "Yontem":        vocab_method,
                "Algoritma":     name,
                "Accuracy":      acc,
                "Precision":     round(precision_score(y_test, y_pred,
                                                       average="weighted",
                                                       zero_division=0), 4),
                "Recall":        round(recall_score(y_test, y_pred,
                                                    average="weighted",
                                                    zero_division=0), 4),
                "F1_Score":      f1,
            })

            print(f"      ✔ {name:<18} | {vocab_method:<4} | "
                  f"Acc={acc:.4f}  F1={f1:.4f}")

    return results
