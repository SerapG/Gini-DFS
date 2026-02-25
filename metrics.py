"""
metrics.py
──────────
Gini İndeksi (GI) ve DFS metrik hesaplama modülü.

Dışa açılan fonksiyonlar:
  build_stats(df)            → Kelime × Sınıf frekans istatistiklerini derler.
  compute_gini(term, stats)  → GI skorunu hesaplar.
  compute_dfs(term, stats)   → DFS skorunu hesaplar.

Formüller
─────────
  GI(t)  = Σ_j  P(t|Cj) · P(Cj|t)
  DFS(t) = Σ_j  P(Cj|t) / ( P(t̄|Cj) + P(t|C̄j) + 1 )

Sıfıra bölme koruması: tüm paydalar + 1e-10 (eps) ile stabilize edilir.
"""

import pandas as pd
from collections import defaultdict

# Sayısal kararlılık için küçük sabit
_EPS = 1e-10


# ─── İSTATİSTİK TABLOSU ───────────────────────────────────────────────────────

def build_stats(df: pd.DataFrame) -> dict:
    """
    Kelime × Sınıf ikili varlık frekans tablosunu oluşturur.

    Her kelime için bir belgede yalnızca BİR kez sayılır (TF değil, binary).

    Döndürülen sözlük alanları:
      classes      : benzersiz sınıf etiketleri   ['ham', 'spam']
      N            : toplam belge sayısı
      class_counts : sınıf → belge sayısı          {'ham': 4827, 'spam': 747}
      term_class   : kelime → sınıf → belge sayısı (defaultdict)
      doc_counts   : kelime → toplam belge sayısı  (defaultdict)

    Parametreler
    ------------
    df : pd.DataFrame  'label' ve 'tokens' sütunları.

    Döndürür
    --------
    dict
    """
    classes      = df["label"].unique().tolist()
    N            = len(df)
    class_counts = df["label"].value_counts().to_dict()

    term_class: dict = defaultdict(lambda: defaultdict(int))
    doc_counts: dict = defaultdict(int)

    for _, row in df.iterrows():
        label = row["label"]
        for token in set(row["tokens"]):     # ikili: her belgede kelimeyi 1 kez say
            term_class[token][label] += 1
            doc_counts[token]        += 1

    return {
        "classes":      classes,
        "N":            N,
        "class_counts": class_counts,
        "term_class":   term_class,
        "doc_counts":   doc_counts,
    }


# ─── GİNİ İNDEKSİ ─────────────────────────────────────────────────────────────

def compute_gini(term: str, stats: dict) -> float:
    """
    Gini İndeksi skorunu hesaplar.

    Formül:
        GI(t) = Σ_j  P(t | Cj) × P(Cj | t)

    Olasılıklar (Bayes):
        P(t | Cj) = n(t,Cj) / n(Cj)     → Cj'de t'yi içeren belge oranı
        P(Cj | t) = n(t,Cj) / n(t)      → t'yi içerenler arasında Cj oranı

    Parametreler
    ------------
    term  : str   Hedef kelime.
    stats : dict  build_stats() çıktısı.

    Döndürür
    --------
    float  GI skoru.
    """
    n_t   = stats["doc_counts"].get(term, 0)
    score = 0.0

    for cls in stats["classes"]:
        n_c   = stats["class_counts"].get(cls, 0)
        n_t_c = stats["term_class"].get(term, {}).get(cls, 0)

        P_t_Cj = n_t_c / (n_c + _EPS)   # P(t | Cj)
        P_Cj_t = n_t_c / (n_t + _EPS)   # P(Cj | t)

        score += P_t_Cj * P_Cj_t

    return score


# ─── DFS ──────────────────────────────────────────────────────────────────────

def compute_dfs(term: str, stats: dict) -> float:
    """
    Differential Feature Selection (DFS) skorunu hesaplar.

    Formül:
        DFS(t) = Σ_j  P(Cj | t) / ( P(t̄ | Cj) + P(t | C̄j) + 1 )

    Olasılıklar (Bayes):
        P(Cj | t)  = n(t,Cj) / n(t)                  → t içerenlerde Cj oranı
        P(t̄ | Cj) = 1 - P(t | Cj)                   → Cj'de t'siz belge oranı
        P(t | C̄j) = (n_t - n_t_c) / (N - n_c)       → Cj dışında t'yi içeren oran

    Paydadaki +1 sabiti sıfıra bölmeyi önler ve skoru (0, 1) aralığında tutar.

    Parametreler
    ------------
    term  : str   Hedef kelime.
    stats : dict  build_stats() çıktısı.

    Döndürür
    --------
    float  DFS skoru.
    """
    N     = stats["N"]
    n_t   = stats["doc_counts"].get(term, 0)
    score = 0.0

    for cls in stats["classes"]:
        n_c   = stats["class_counts"].get(cls, 0)
        n_t_c = stats["term_class"].get(term, {}).get(cls, 0)

        P_Cj_t     = n_t_c / (n_t + _EPS)                    # P(Cj | t)
        P_not_t_Cj = 1.0 - n_t_c / (n_c + _EPS)             # P(t̄ | Cj)
        P_t_not_Cj = (n_t - n_t_c) / ((N - n_c) + _EPS)     # P(t | C̄j)

        score += P_Cj_t / (P_not_t_Cj + P_t_not_Cj + 1.0)

    return score
