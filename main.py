"""
main.py
───────
Ana modül

Her çalistirmada:
  • 'results/' klasörü yoksa olusturulur.
  • Otomatik çalistirma numarasi belirlenir.
  • Gini/DFS skor matrisleri (en sik 500 kelime) CSV'ye kaydedilir.
  • VOCAB_SIZES listesindeki her boyut için TF-IDF + SVM/MNB/RF deneyleri yapilir.
  • Tüm sonuçlar tek bir 'final_comparison_results.csv' dosyasinda toplanir.

Çalistirma:
  python main.py
"""

import os
import sys
import glob
import pandas as pd
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data_loader  import load_english, load_turkish
from metrics      import build_stats, compute_gini, compute_dfs
from classifier   import prepare_data, run_vocab_experiment
import view_results

# ─── YAPILANDIRMA ─────────────────────────────────────────────────────────────

# Skor matrisi için sabit (en sik 500 kelime skorlanir)
SCORE_TOP_N  = 500

# Siniflandirma deneyleri için denenecek vocabulary boyutlari
VOCAB_SIZES  = [500, 300, 100, 50, 30, 10]

RESULTS_DIR  = "results"


# ─── ÇALIŞMA NUMARASI ─────────────────────────────────────────────────────────

def _next_run_number() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    nums = set()
    for path in existing:
        base = os.path.basename(path)
        part = base.split("-", 1)[0]
        if part.isdigit():
            nums.add(int(part))
    return max(nums) + 1 if nums else 1


# ─── SKOR MATRİSİ ─────────────────────────────────────────────────────────────

def build_score_matrix(df: pd.DataFrame, lang_label: str) -> pd.DataFrame:
    """
    En sik kullanilan SCORE_TOP_N kelime için Gini ve DFS skorlarini hesaplar.
    Döndürülen sütunlar: Kelime | Frekans | Gini_Score | DFS_Score
    """
    print(f"  [{lang_label}] Istatistikler hesaplaniyor...")
    stats     = build_stats(df)
    all_terms = list(stats["term_class"].keys())
    print(f"  → Benzersiz kelime  : {len(all_terms)}")

    freq: Counter = Counter()
    for tokens in df["tokens"]:
        freq.update(tokens)

    top_terms = [term for term, _ in freq.most_common(SCORE_TOP_N)]
    print(f"  → En sik {SCORE_TOP_N} kelime secildi (toplam token: {sum(freq.values())})")

    rows = [
        {
            "Kelime":     term,
            "Frekans":    freq[term],
            "Gini_Score": round(compute_gini(term, stats), 6),
            "DFS_Score":  round(compute_dfs(term, stats),  6),
        }
        for term in top_terms
    ]

    result = (
        pd.DataFrame(rows)
        .sort_values("Gini_Score", ascending=False)
        .reset_index(drop=True)
    )
    result.index     += 1
    result.index.name = "Sira"
    return result


def display_comparison(matrix: pd.DataFrame, lang_label: str) -> None:
    """İlk 10 satiri önizleme olarak ekrana basar."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {lang_label} – En Sik {SCORE_TOP_N} Kelime | Gini vs DFS (ilk 10)")
    print(sep)
    print(matrix.head(10).to_string())
    print(f"  ... ({len(matrix)} satir, tamami CSV'de)\n")


def save(matrix: pd.DataFrame, output_path: str) -> None:
    """Matrisi UTF-8 BOM ile CSV olarak kaydeder."""
    matrix.to_csv(output_path, index=True, encoding="utf-8-sig")
    print(f"  ✓ Kaydedildi → {output_path}")


# ─── DİL PİPELINE'LARI ────────────────────────────────────────────────────────

def run_english(run_no: int) -> list:
    """
    Ingilizce SMS pipeline:
      Yukle → Skorla → CSV kaydet → Split+Train/Test CSV → Vocab döngüsü
    Tüm deneme satirlarini döndürür.
    """
    print(f"\n{'─' * 62}")
    print("  [Ingilizce SMS] okunuyor...")
    df = load_english()
    print(f"  → Toplam mesaj      : {len(df)}")
    print(f"  → Sinif dagilimi    :\n{df['label'].value_counts().to_string()}")

    # Skor matrisi (500 kelime skorlanir, CSV'ye kaydedilir)
    matrix      = build_score_matrix(df, "Ingilizce SMS")
    output_path = os.path.join(RESULTS_DIR, f"{run_no}-english_analysis.csv")
    display_comparison(matrix, "Ingilizce SMS")
    save(matrix, output_path)

    # Train/Test bölme ve CSV kayit (1 kez)
    train_path, test_path = prepare_data(df, "english", run_no)

    # Vocab boyutlari üzerinden döngü
    all_rows = []
    for vocab_size in VOCAB_SIZES:
        print(f"\n  >>> Analiz ediliyor: {vocab_size} kelime... (Ingilizce)")
        rows = run_vocab_experiment(
            train_path, test_path, matrix, vocab_size, "Ingilizce"
        )
        all_rows.extend(rows)

    return all_rows


def run_turkish(run_no: int) -> list:
    """
    Türkçe SMS pipeline:
      Yukle → Skorla → CSV kaydet → Split+Train/Test CSV → Vocab döngüsü
    Tüm deneme satirlarini döndürür.
    """
    print(f"\n{'─' * 62}")
    print("  [Turkce SMS] os.walk ile taraniyor...")
    df = load_turkish()
    print(f"  → Toplam mesaj      : {len(df)}")
    print(f"  → Sinif dagilimi    :\n{df['label'].value_counts().to_string()}")

    # Skor matrisi
    matrix      = build_score_matrix(df, "Turkce SMS")
    output_path = os.path.join(RESULTS_DIR, f"{run_no}-turkish_analysis.csv")
    display_comparison(matrix, "Turkce SMS")
    save(matrix, output_path)

    # Train/Test bölme ve CSV kayit (1 kez)
    train_path, test_path = prepare_data(df, "turkish", run_no)

    # Vocab boyutlari üzerinden döngü
    all_rows = []
    for vocab_size in VOCAB_SIZES:
        print(f"\n  >>> Analiz ediliyor: {vocab_size} kelime... (Turkce)")
        rows = run_vocab_experiment(
            train_path, test_path, matrix, vocab_size, "Turkce"
        )
        all_rows.extend(rows)

    return all_rows


# ─── ANA GİRİŞ NOKTASI ────────────────────────────────────────────────────────

def main():
    run_no = _next_run_number()

    print("=" * 62)
    print("  Oz Nitelik Secimi: Gini Indeksi & DFS + ML Siniflandirma")
    print(f"  Calistirma #{run_no}  |  Vocab boyutlari: {VOCAB_SIZES}")
    print("=" * 62)

    # Her iki dil için pipeline'lari çalistir, satirlari topla
    all_rows  = run_english(run_no)
    all_rows += run_turkish(run_no)

    # ── Kümülatif final tablosunu oluştur ve kaydet ───────────────────────────
    final_df = pd.DataFrame(all_rows, columns=[
        "Dil", "Kelime_Sayisi", "Yontem", "Algoritma",
        "Accuracy", "Precision", "Recall", "F1_Score",
    ])

    final_path = os.path.join(RESULTS_DIR, f"{run_no}-final_comparison_results.csv")
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")

    # ── Özet konsol çıktısı ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  OZET – F1 Skorlari (Kelime Sayisina Gore)")
    print("=" * 62)
    pivot = (
        final_df
        .groupby(["Dil", "Kelime_Sayisi", "Yontem", "Algoritma"])["F1_Score"]
        .first()
        .unstack(["Yontem", "Algoritma"])
    )
    print(pivot.to_string())

    print(f"\n✓ Tum analizler tamamlandi.")
    print(f"  → Ingilizce analiz     : {RESULTS_DIR}/{run_no}-english_analysis.csv")
    print(f"  → Turkce analiz        : {RESULTS_DIR}/{run_no}-turkish_analysis.csv")
    print(f"  → Final karsilastirma  : {final_path}")
    print(f"     ({len(final_df)} satir: "
          f"{len(VOCAB_SIZES)} boyut x 2 dil x 3 algo x 2 yontem)")

    # Analizler tamamlandiktan sonra sonuclari tarayicide ac
    print("\nSonuclar tarayicide aciliyor...")
    view_results.main()


if __name__ == "__main__":
    main()
