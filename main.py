"""
main.py
───────
Ana modül

Her çalıştırmada:
  • 'results/' klasörü yoksa oluşturulur.
  • Bir sonraki çalıştırma numarasını (1, 2, 3 …) otomatik belirler.
  • Çıktılar  results/1-english_analysis.csv  ve  results/1-turkish_analysis.csv
    biçiminde kaydedilir.

Çalıştırma:
  python main.py
"""

import os
import glob
import pandas as pd

from data_loader import load_english, load_turkish
from metrics     import build_stats, compute_gini, compute_dfs

# ─── YAPILANDIRMA ─────────────────────────────────────────────────────────────

TOP_N          = 20           # raporlanacak kelime sayısı
RESULTS_DIR    = "results"    # tüm çıktı CSV dosyaları bu klasöre kaydedilir


# ─── ÇALIŞMA NUMARASI ─────────────────────────────────────────────────────────

def _next_run_number() -> int:
    """
    results/ klasöründeki mevcut dosyaların numaralarına bakarak
    bir sonraki çalıştırma numarasını döndürür.

    Örnek:
      results/ boşsa          → 1
      1-english… varsa        → 2
      1-…, 2-… varsa         → 3

    Döndürür
    --------
    int  Bir sonraki çalıştırma numarası.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)   # klasörü oluştur (yoksa)

    # Mevcut numaraları topla
    existing = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    nums = set()
    for path in existing:
        base = os.path.basename(path)
        # Beklenen biçim: "<N>-*.csv"
        part = base.split("-", 1)[0]
        if part.isdigit():
            nums.add(int(part))

    return max(nums) + 1 if nums else 1


# ─── ANALİZ: SKORLAMA + ANA MATRİS ───────────────────────────────────────────

def build_score_matrix(df: pd.DataFrame, lang_label: str) -> pd.DataFrame:
    """
    Veri setindeki her kelime için GI ve DFS skorlarını hesaplar.
    Gini_Score'a göre azalan sıralanmış Top-N ana matrisi döndürür.

    Döndürülen sütunlar: Kelime | Gini_Score | DFS_Score

    Parametreler
    ------------
    df         : pd.DataFrame  'label' ve 'tokens' sütunları.
    lang_label : str           Ekran çıktısı için dil etiketi.

    Döndürür
    --------
    pd.DataFrame
    """
    print(f"  [{lang_label}] İstatistikler hesaplanıyor…")
    stats     = build_stats(df)
    all_terms = list(stats["term_class"].keys())
    print(f"  → Benzersiz kelime  : {len(all_terms)}")

    rows = [
        {
            "Kelime":     term,
            "Gini_Score": round(compute_gini(term, stats), 6),
            "DFS_Score":  round(compute_dfs(term, stats),  6),
        }
        for term in all_terms
    ]

    top = (
        pd.DataFrame(rows)
        .sort_values("Gini_Score", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    top.index     += 1
    top.index.name = "Sıra"
    return top


# ─── KARŞILAŞTIRMALI ÇIKTI ────────────────────────────────────────────────────

def display_comparison(matrix: pd.DataFrame, lang_label: str) -> None:
    """GI ve DFS sütunlarını yan yana karşılaştırmalı tablo olarak ekrana basar."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {lang_label} – Top {TOP_N} Kelime | Gini  vs  DFS")
    print(sep)
    print(matrix.to_string())
    print()


def save(matrix: pd.DataFrame, output_path: str) -> None:
    """Matrisi UTF-8 BOM ile CSV olarak results/ klasörüne kaydeder."""
    matrix.to_csv(output_path, index=True, encoding="utf-8-sig")
    print(f"  ✓ Kaydedildi → {output_path}")


# ─── PIPELINE'LAR ─────────────────────────────────────────────────────────────

def run_english(run_no: int) -> pd.DataFrame:
    """İngilizce SMS pipeline: yükle → skorla → göster → kaydet."""
    print(f"\n{'─' * 62}")
    print(f"  [İngilizce SMS] okunuyor…")
    df = load_english()
    print(f"  → Toplam mesaj      : {len(df)}")
    print(f"  → Sınıf dağılımı    :\n{df['label'].value_counts().to_string()}")

    matrix      = build_score_matrix(df, "İngilizce SMS")
    output_path = os.path.join(RESULTS_DIR, f"{run_no}-english_analysis.csv")

    display_comparison(matrix, "İngilizce SMS")
    save(matrix, output_path)
    return matrix


def run_turkish(run_no: int) -> pd.DataFrame:
    """Türkçe SMS pipeline: yükle → skorla → göster → kaydet."""
    print(f"\n{'─' * 62}")
    print(f"  [Türkçe SMS] os.walk ile taranıyor…")
    df = load_turkish()
    print(f"  → Toplam mesaj      : {len(df)}")
    print(f"  → Sınıf dağılımı    :\n{df['label'].value_counts().to_string()}")

    matrix      = build_score_matrix(df, "Türkçe SMS")
    output_path = os.path.join(RESULTS_DIR, f"{run_no}-turkish_analysis.csv")

    display_comparison(matrix, "Türkçe SMS")
    save(matrix, output_path)
    return matrix


# ─── ANA GİRİŞ NOKTASI ────────────────────────────────────────────────────────

def main():
    run_no = _next_run_number()   # bu çalıştırmaya ait sıra numarası

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Öz Nitelik Seçimi: Gini İndeksi & DFS Analizi         ║")
    print(f"║   Çalıştırma #{run_no:<49}║")
    print("╚══════════════════════════════════════════════════════════╝")

    run_english(run_no)
    run_turkish(run_no)

    print("\n✅ Tüm analizler tamamlandı.")
    print(f"   → İngilizce çıktı : {RESULTS_DIR}/{run_no}-english_analysis.csv")
    print(f"   → Türkçe çıktı    : {RESULTS_DIR}/{run_no}-turkish_analysis.csv")


if __name__ == "__main__":
    main()
