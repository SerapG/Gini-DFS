"""
view_results.py
───────────────
En son calistirmaya ait tüm analiz dosyalarini tarayicida gösterir.

Sekmeler:
  📊 Final Karsilastirma  – final_comparison_results.csv (pivot + flat tablo)
  🇬🇧 Ingilizce           – Gini/DFS skor tablosu
  🇹🇷 Türkce              – Gini/DFS skor tablosu
  🗂  Veri Seti            – Train/Test bölünme istatistikleri

Calistirma:
  python view_results.py
"""

import os
import sys
import glob
import webbrowser
import tempfile
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = "results"


# ─── Yardımcı: En son çalıştırma numarası ─────────────────────────────────────

def _latest_run_number():
    existing = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    nums = set()
    for path in existing:
        base = os.path.basename(path)
        part = base.split("-", 1)[0]
        if part.isdigit():
            nums.add(int(part))
    return max(nums) if nums else None


# ─── Gini/DFS Tablo ───────────────────────────────────────────────────────────

def _df_to_html_table(df: pd.DataFrame) -> str:
    rows_html = ""
    max_gini = df["Gini_Score"].max() if "Gini_Score" in df.columns else 1
    for idx, row in df.iterrows():
        bar_w = max(2, int(row["Gini_Score"] / max_gini * 80))
        rows_html += (
            f"<tr>"
            f"<td class='rank'>{idx}</td>"
            f"<td class='word'>{row['Kelime']}</td>"
            f"<td class='num'>{int(row['Frekans']):,}</td>"
            f"<td class='score'>{row['Gini_Score']:.6f}"
            f"<span class='bar gini-bar' style='width:{bar_w}px'></span></td>"
            f"<td class='score'>{row['DFS_Score']:.6f}</td>"
            f"</tr>\n"
        )
    return rows_html


# ─── Train/Test İstatistik Satırları ──────────────────────────────────────────

def _split_stats_html(train_path, test_path):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return "<tr><td colspan='4' style='color:#64748b;text-align:center'>CSV bulunamadi</td></tr>"
    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test  = pd.read_csv(test_path,  encoding="utf-8-sig")

    def rows(df, split):
        counts = df["label"].value_counts()
        total  = len(df)
        html   = ""
        for label, cnt in counts.items():
            c = "#f87171" if label == "spam" else "#34d399"
            html += (
                f"<tr>"
                f"<td style='color:#94a3b8'>{split}</td>"
                f"<td style='color:{c};font-weight:600'>{label}</td>"
                f"<td class='num'>{cnt:,}</td>"
                f"<td class='num'>{cnt/total*100:.1f}%</td>"
                f"</tr>\n"
            )
        return html
    return rows(df_train, "Train") + rows(df_test, "Test")


# ─── Final Karşılaştırma: Pivot Isı Haritası ──────────────────────────────────

def _f1_to_color(val: float, vmin: float, vmax: float) -> str:
    """F1 skorunu yeşil tonunda bir arka plan rengine çevirir."""
    if vmax == vmin:
        t = 0.5
    else:
        t = (val - vmin) / (vmax - vmin)
    # düşük: kırmızımsı, yüksek: yeşilimsi (dark mode uyumlu)
    r = int(220 * (1 - t) + 30  * t)
    g = int(60  * (1 - t) + 200 * t)
    b = int(60  * (1 - t) + 90  * t)
    brightness = 0.299*r + 0.587*g + 0.114*b
    text = "#0f1117" if brightness > 120 else "#e2e8f0"
    return f"background:rgb({r},{g},{b});color:{text}"


def _final_pivot_html(df_final: pd.DataFrame, dil: str) -> str:
    """Belirli dil için pivot tablo (satır=Kelime_Sayisi, kolon=Algo×Yöntem)."""
    sub = df_final[df_final["Dil"] == dil].copy()
    if sub.empty:
        return "<p style='color:#64748b'>Veri yok</p>"

    vocab_sizes = sorted(sub["Kelime_Sayisi"].unique(), reverse=True)
    combos      = [
        (algo, yontem)
        for yontem in ["Gini", "DFS"]
        for algo in ["SVM", "MNB", "Random Forest"]
    ]

    vmin = sub["F1_Score"].min()
    vmax = sub["F1_Score"].max()

    # Başlık
    header  = "<thead>"
    header += "<tr>"
    header += "<th rowspan='2' style='text-align:center;vertical-align:middle'>Kelime<br>Sayisi</th>"
    header += "<th colspan='3' style='text-align:center;border-left:1px solid #2d3148'>Gini Vocabulary</th>"
    header += "<th colspan='3' style='text-align:center;border-left:2px solid #6366f1'>DFS Vocabulary</th>"
    header += "</tr><tr>"
    for yontem in ["Gini", "DFS"]:
        border = "border-left:1px solid #2d3148" if yontem == "Gini" else "border-left:2px solid #6366f1"
        for algo in ["SVM", "MNB", "Random Forest"]:
            header += f"<th style='text-align:center;{border}'>{algo}</th>"
            border = ""
    header += "</tr></thead>"

    # Satırlar
    body = "<tbody>"
    for vs in vocab_sizes:
        body += f"<tr><td class='rank' style='font-size:.9rem;font-weight:600;color:#a5b4fc'>{vs}</td>"
        first_dfs = True
        for yontem in ["Gini", "DFS"]:
            for algo in ["SVM", "MNB", "Random Forest"]:
                mask = (sub["Kelime_Sayisi"] == vs) & (sub["Yontem"] == yontem) & (sub["Algoritma"] == algo)
                row  = sub[mask]
                if row.empty:
                    body += "<td style='text-align:center;color:#475569'>—</td>"
                else:
                    f1    = float(row["F1_Score"].iloc[0])
                    acc   = float(row["Accuracy"].iloc[0])
                    style = _f1_to_color(f1, vmin, vmax)
                    left  = "border-left:2px solid #6366f1;" if (yontem == "DFS" and first_dfs) else ""
                    first_dfs = False
                    body += (
                        f"<td style='text-align:center;{style};{left}padding:8px 6px' "
                        f"title='Acc={acc:.4f} | F1={f1:.4f}'>"
                        f"<span style='font-weight:700;font-size:.875rem'>{f1:.4f}</span>"
                        f"<br><span style='font-size:.7rem;opacity:.7'>Acc {acc:.3f}</span>"
                        f"</td>"
                    )
        body += "</tr>\n"
    body += "</tbody>"
    return f"<table class='pivot-table'>{header}{body}</table>"


def _final_flat_html(df_final: pd.DataFrame) -> str:
    """72 satırlık flat tablo."""
    algo_colors = {"SVM": "#6366f1", "MNB": "#06b6d4", "Random Forest": "#10b981"}
    vmin = df_final["F1_Score"].min()
    vmax = df_final["F1_Score"].max()

    rows_html = ""
    for _, row in df_final.iterrows():
        algo  = row["Algoritma"]
        color = algo_colors.get(algo, "#94a3b8")
        f1    = float(row["F1_Score"])
        bar_w = max(2, int((f1 - vmin) / (vmax - vmin + 1e-9) * 80))
        dil_text = str(row["Dil"])
        vocab_badge = (
            "<span class='badge-gini'>Gini</span>"
            if row["Yontem"] == "Gini"
            else "<span class='badge-dfs'>DFS</span>"
        )
        rows_html += (
            f"<tr>"
            f"<td style='text-align:center;font-weight:600;color:#a5b4fc'>{dil_text}</td>"
            f"<td style='color:#a5b4fc;font-weight:600;text-align:center'>{int(row['Kelime_Sayisi'])}</td>"
            f"<td style='text-align:center'>{vocab_badge}</td>"
            f"<td style='color:{color};font-weight:600'>{algo}</td>"
            f"<td class='metric'>{float(row['Accuracy']):.4f}</td>"
            f"<td class='metric'>{float(row['Precision']):.4f}</td>"
            f"<td class='metric'>{float(row['Recall']):.4f}</td>"
            f"<td class='metric'>{f1:.4f}"
            f"<span class='bar acc-bar' style='width:{bar_w}px'></span></td>"
            f"</tr>\n"
        )
    return rows_html


# ─── HTML Üretici ─────────────────────────────────────────────────────────────

def generate_html(run_no: int) -> str:
    eng_path    = os.path.join(RESULTS_DIR, f"{run_no}-english_analysis.csv")
    tur_path    = os.path.join(RESULTS_DIR, f"{run_no}-turkish_analysis.csv")
    final_path  = os.path.join(RESULTS_DIR, f"{run_no}-final_comparison_results.csv")
    train_eng   = os.path.join(RESULTS_DIR, f"{run_no}-english_train.csv")
    test_eng    = os.path.join(RESULTS_DIR, f"{run_no}-english_test.csv")
    train_tur   = os.path.join(RESULTS_DIR, f"{run_no}-turkish_train.csv")
    test_tur    = os.path.join(RESULTS_DIR, f"{run_no}-turkish_test.csv")

    df_eng = pd.read_csv(eng_path, encoding="utf-8-sig", index_col=0)
    df_tur = pd.read_csv(tur_path, encoding="utf-8-sig", index_col=0)
    df_eng.index.name = "Sira"
    df_tur.index.name = "Sira"
    eng_rows = _df_to_html_table(df_eng)
    tur_rows = _df_to_html_table(df_tur)
    total_eng = int(df_eng["Frekans"].sum())
    total_tur = int(df_tur["Frekans"].sum())

    # Final karşılaştırma
    has_final = os.path.exists(final_path)
    if has_final:
        df_final = pd.read_csv(final_path, encoding="utf-8-sig")
        pivot_eng_html = _final_pivot_html(df_final, "Ingilizce")
        pivot_tur_html = _final_pivot_html(df_final, "Turkce")
        flat_html      = _final_flat_html(df_final)
        total_exp      = len(df_final)
        best_row       = df_final.loc[df_final["F1_Score"].idxmax()]
        best_info      = f"{best_row['Algoritma']} / {best_row['Yontem']} / {int(best_row['Kelime_Sayisi'])} kelime"
    else:
        pivot_eng_html = pivot_tur_html = flat_html = "<p style='color:#64748b;text-align:center;padding:40px'>final_comparison_results.csv bulunamadi.</p>"
        total_exp = 0
        best_info = "—"

    split_eng = _split_stats_html(train_eng, test_eng)
    split_tur = _split_stats_html(train_tur, test_tur)

    best_f1_str = f"{df_final['F1_Score'].max():.4f}" if has_final else "—"

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Gini &amp; DFS Analizi – #{run_no}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:28px 20px}}
  header{{text-align:center;margin-bottom:32px}}
  header h1{{font-size:1.75rem;font-weight:700;background:linear-gradient(135deg,#6366f1,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:6px}}
  header p{{color:#64748b;font-size:.875rem}}

  .tabs{{display:flex;justify-content:center;gap:8px;margin-bottom:28px;flex-wrap:wrap}}
  .tab-btn{{padding:9px 22px;border:none;border-radius:999px;font-family:inherit;font-size:.85rem;
    font-weight:600;cursor:pointer;transition:all .2s;background:#1e2130;color:#64748b}}
  .tab-btn.active{{background:linear-gradient(135deg,#6366f1,#06b6d4);color:#fff;box-shadow:0 4px 15px rgba(99,102,241,.4)}}
  .panel{{display:none}}.panel.active{{display:block;animation:fadeIn .3s ease}}
  @keyframes fadeIn{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}

  .stats-bar{{display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap}}
  .stat-card{{flex:1;min-width:120px;background:#1e2130;border:1px solid #2d3148;border-radius:12px;padding:13px 16px}}
  .stat-card .label{{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
  .stat-card .value{{font-size:1.2rem;font-weight:700;color:#e2e8f0}}

  .sub-tabs{{display:flex;gap:8px;margin-bottom:16px}}
  .sub-btn{{padding:6px 18px;border:1px solid #2d3148;border-radius:999px;font-family:inherit;
    font-size:.8rem;font-weight:600;cursor:pointer;background:#1e2130;color:#64748b;transition:all .2s}}
  .sub-btn.active{{border-color:#6366f1;color:#a5b4fc;background:#1e1f38}}
  .sub-panel{{display:none}}.sub-panel.active{{display:block}}

  .search-box{{margin-bottom:12px}}
  .search-box input{{width:100%;max-width:320px;padding:8px 14px;background:#1e2130;border:1px solid #2d3148;
    border-radius:8px;color:#e2e8f0;font-family:inherit;font-size:.85rem;outline:none;transition:border-color .2s}}
  .search-box input:focus{{border-color:#6366f1}}
  .search-box input::placeholder{{color:#475569}}

  .table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid #2d3148;max-height:560px;overflow-y:auto}}
  .table-wrap::-webkit-scrollbar{{width:5px;height:5px}}
  .table-wrap::-webkit-scrollbar-track{{background:#1e2130}}
  .table-wrap::-webkit-scrollbar-thumb{{background:#374151;border-radius:3px}}

  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  thead th{{background:#181b2e;padding:11px 14px;text-align:left;font-size:.7rem;font-weight:600;
    text-transform:uppercase;letter-spacing:.06em;color:#64748b;position:sticky;top:0;z-index:1;
    border-bottom:1px solid #2d3148;white-space:nowrap}}
  tbody tr{{border-bottom:1px solid #1e2130;transition:background .15s}}
  tbody tr:hover{{background:#1e2540}}
  tbody tr:last-child{{border-bottom:none}}
  td{{padding:9px 14px;vertical-align:middle}}

  td.rank{{color:#475569;font-size:.8rem;font-weight:500;width:52px;text-align:center}}
  td.word{{font-weight:600;color:#a5b4fc;font-size:.875rem}}
  td.num{{color:#34d399;font-weight:500;text-align:right}}
  td.score,td.metric{{color:#94a3b8;text-align:right;font-variant-numeric:tabular-nums;font-size:.8rem}}
  .bar{{display:inline-block;height:3px;border-radius:2px;margin-left:5px;vertical-align:middle;min-width:2px}}
  .gini-bar{{background:linear-gradient(90deg,#6366f1,#06b6d4)}}
  .acc-bar{{background:linear-gradient(90deg,#10b981,#06b6d4)}}
  .highlight td.word{{color:#fbbf24}}

  /* Pivot table */
  .pivot-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
  .pivot-table thead th{{background:#181b2e;padding:10px 8px;text-align:center;font-size:.68rem;
    font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#64748b;
    position:sticky;top:0;z-index:1;border-bottom:1px solid #2d3148;white-space:nowrap}}
  .pivot-table tbody tr{{border-bottom:1px solid #1a1d2e}}
  .pivot-table tbody tr:hover td{{filter:brightness(1.15)}}
  .pivot-table td{{padding:0;transition:filter .15s}}

  .lang-toggle{{display:flex;gap:8px;margin-bottom:14px}}
  .lang-btn{{padding:7px 20px;border:1px solid #2d3148;border-radius:999px;font-family:inherit;
    font-size:.82rem;font-weight:600;cursor:pointer;background:#1e2130;color:#64748b;transition:all .2s}}
  .lang-btn.active{{border-color:#6366f1;color:#a5b4fc;background:#1e1f38}}
  .lang-panel{{display:none}}.lang-panel.active{{display:block}}

  .badge-gini{{display:inline-block;padding:2px 9px;border-radius:999px;background:#312e81;color:#a5b4fc;font-size:.7rem;font-weight:600}}
  .badge-dfs{{display:inline-block;padding:2px 9px;border-radius:999px;background:#164e63;color:#67e8f9;font-size:.7rem;font-weight:600}}
  .section-title{{font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin:22px 0 10px}}

  .legend{{display:flex;align-items:center;gap:12px;margin-bottom:12px;font-size:.75rem;color:#64748b}}
  .legend-grad{{width:120px;height:10px;border-radius:4px;background:linear-gradient(90deg,rgb(220,60,60),rgb(30,200,90));display:inline-block}}
</style>
</head>
<body>

<header>
  <h1>📊 Gini &amp; DFS + ML Siniflandirma Analizi</h1>
  <p>Calistirma #{run_no} &nbsp;·&nbsp; Vocabulary: [500, 300, 100, 50, 30, 10] &nbsp;·&nbsp; SVM · MNB · Random Forest</p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('final',this)">📊 Final Karsilastirma</button>
  <button class="tab-btn"        onclick="switchTab('eng',  this)">🇬🇧 Ingilizce SMS</button>
  <button class="tab-btn"        onclick="switchTab('tur',  this)">🇹🇷 Turkce SMS</button>
  <button class="tab-btn"        onclick="switchTab('ds',   this)">🗂 Veri Seti</button>
</div>

<!-- ══ FINAL KARŞILAŞTIRMA ══════════════════════════════════════════════════ -->
<div id="panel-final" class="panel active">

  <div class="stats-bar">
    <div class="stat-card"><div class="label">Toplam Deney</div><div class="value">{total_exp}</div></div>
    <div class="stat-card"><div class="label">Vocab Boyutlari</div><div class="value">6</div></div>
    <div class="stat-card"><div class="label">En Yuksek F1<br><span style="font-size:.65rem;font-weight:400;color:#94a3b8">{best_info}</span></div>
      <div class="value" style="color:#34d399">{best_f1_str}</div></div>
    <div class="stat-card"><div class="label">Algoritmalar</div><div class="value">SVM · MNB · RF</div></div>
  </div>

  <!-- Alt sekme: Pivot / Flat -->
  <div class="sub-tabs">
    <button class="sub-btn active" onclick="switchSub('pivot',this)">🌡 Isı Haritası (Pivot)</button>
    <button class="sub-btn"       onclick="switchSub('flat', this)">📋 Tüm Satırlar</button>
  </div>

  <!-- Pivot görünümü -->
  <div id="sub-pivot" class="sub-panel active">
    <div class="legend">
      <span>Düsük F1</span>
      <span class="legend-grad"></span>
      <span>Yüksek F1</span>
      <span style="margin-left:16px">— Hücreye fareyle gel: Acc + F1 detay</span>
    </div>
    <div class="lang-toggle">
      <button class="lang-btn active" onclick="switchLang('peng',this)">🇬🇧 Ingilizce</button>
      <button class="lang-btn"        onclick="switchLang('ptur',this)">🇹🇷 Turkce</button>
    </div>
    <div id="lang-peng" class="lang-panel active">
      <div class="table-wrap">{pivot_eng_html}</div>
    </div>
    <div id="lang-ptur" class="lang-panel">
      <div class="table-wrap">{pivot_tur_html}</div>
    </div>
  </div>

  <!-- Flat tablo -->
  <div id="sub-flat" class="sub-panel">
    <div class="search-box">
      <input type="text" id="search-flat" placeholder="Algoritma / Yontem / Dil ara..." oninput="filterFlat(this.value)">
    </div>
    <div class="table-wrap">
      <table id="table-flat">
        <thead><tr>
          <th style="text-align:center">Dil</th>
          <th style="text-align:center">Kelime<br>Sayisi</th>
          <th style="text-align:center">Yontem</th>
          <th>Algoritma</th>
          <th style="text-align:right">Accuracy</th>
          <th style="text-align:right">Precision</th>
          <th style="text-align:right">Recall</th>
          <th style="text-align:right">F1 Score ▼</th>
        </tr></thead>
        <tbody>{flat_html}</tbody>
      </table>
    </div>
  </div>
</div>

<!-- ══ İNGİLİZCE ════════════════════════════════════════════════════════════ -->
<div id="panel-eng" class="panel">
  <div class="stats-bar">
    <div class="stat-card"><div class="label">Analiz Edilen Kelime</div><div class="value">{len(df_eng)}</div></div>
    <div class="stat-card"><div class="label">Toplam Token</div><div class="value">{total_eng:,}</div></div>
    <div class="stat-card"><div class="label">Max Gini</div><div class="value">{df_eng['Gini_Score'].max():.4f}</div></div>
    <div class="stat-card"><div class="label">Max DFS</div><div class="value">{df_eng['DFS_Score'].max():.4f}</div></div>
  </div>
  <div class="search-box">
    <input type="text" id="search-eng" placeholder="Kelime ara..." oninput="filterTable('eng',this.value)">
  </div>
  <div class="table-wrap">
    <table id="table-eng">
      <thead><tr>
        <th style="text-align:center">Sira</th><th>Kelime</th>
        <th style="text-align:right">Frekans</th>
        <th style="text-align:right">Gini Score ▼</th>
        <th style="text-align:right">DFS Score</th>
      </tr></thead>
      <tbody>{eng_rows}</tbody>
    </table>
  </div>
</div>

<!-- ══ TÜRKÇE ════════════════════════════════════════════════════════════════ -->
<div id="panel-tur" class="panel">
  <div class="stats-bar">
    <div class="stat-card"><div class="label">Analiz Edilen Kelime</div><div class="value">{len(df_tur)}</div></div>
    <div class="stat-card"><div class="label">Toplam Token</div><div class="value">{total_tur:,}</div></div>
    <div class="stat-card"><div class="label">Max Gini</div><div class="value">{df_tur['Gini_Score'].max():.4f}</div></div>
    <div class="stat-card"><div class="label">Max DFS</div><div class="value">{df_tur['DFS_Score'].max():.4f}</div></div>
  </div>
  <div class="search-box">
    <input type="text" id="search-tur" placeholder="Kelime ara..." oninput="filterTable('tur',this.value)">
  </div>
  <div class="table-wrap">
    <table id="table-tur">
      <thead><tr>
        <th style="text-align:center">Sira</th><th>Kelime</th>
        <th style="text-align:right">Frekans</th>
        <th style="text-align:right">Gini Score ▼</th>
        <th style="text-align:right">DFS Score</th>
      </tr></thead>
      <tbody>{tur_rows}</tbody>
    </table>
  </div>
</div>

<!-- ══ VERİ SETİ ═════════════════════════════════════════════════════════════ -->
<div id="panel-ds" class="panel">
  <p class="section-title">🇬🇧 Ingilizce – Train / Test (%70 / %30)</p>
  <div class="table-wrap" style="margin-bottom:24px">
    <table><thead><tr>
      <th>Bölüm</th><th>Etiket</th>
      <th style="text-align:right">Belge</th>
      <th style="text-align:right">Oran</th>
    </tr></thead><tbody>{split_eng}</tbody></table>
  </div>
  <p class="section-title">🇹🇷 Turkce – Train / Test (%70 / %30)</p>
  <div class="table-wrap">
    <table><thead><tr>
      <th>Bölüm</th><th>Etiket</th>
      <th style="text-align:right">Belge</th>
      <th style="text-align:right">Oran</th>
    </tr></thead><tbody>{split_tur}</tbody></table>
  </div>
</div>

<script>
function switchTab(id,btn){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  btn.classList.add('active');
}}
function switchSub(id,btn){{
  document.querySelectorAll('.sub-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.sub-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('sub-'+id).classList.add('active');
  btn.classList.add('active');
}}
function switchLang(id,btn){{
  btn.closest('.sub-panel').querySelectorAll('.lang-panel').forEach(p=>p.classList.remove('active'));
  btn.closest('.sub-panel').querySelectorAll('.lang-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('lang-'+id).classList.add('active');
  btn.classList.add('active');
}}
function filterTable(lang,q){{
  q=q.toLowerCase().trim();
  document.querySelectorAll('#table-'+lang+' tbody tr').forEach(row=>{{
    const w=row.querySelector('td.word')?.textContent.toLowerCase()??'';
    const show=!q||w.includes(q);
    row.style.display=show?'':'none';
    row.classList.toggle('highlight',show&&q.length>0);
  }});
}}
function filterFlat(q){{
  q=q.toLowerCase().trim();
  document.querySelectorAll('#table-flat tbody tr').forEach(row=>{{
    const txt=row.textContent.toLowerCase();
    row.style.display=(!q||txt.includes(q))?'':'none';
  }});
}}
</script>
</body>
</html>"""
    return html


def main():
    run_no = _latest_run_number()
    if run_no is None:
        print("results/ klasorunde hic CSV bulunamadi. Once 'python main.py' calistirin.")
        return

    print(f"Calistirma #{run_no} sonuclari yukleniyor...")
    html_content = generate_html(run_no)

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8", prefix="gini_dfs_"
    )
    tmp.write(html_content)
    tmp.close()

    print(f"HTML olusturuldu -> {tmp.name}")
    webbrowser.open(f"file:///{tmp.name}")
    print("Tarayici aciliyor...")


if __name__ == "__main__":
    main()
