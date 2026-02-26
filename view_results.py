"""
view_results.py
───────────────
En son çalıştırmaya ait English ve Turkish analiz CSV dosyalarını
güzel biçimlendirilmiş HTML tablolar olarak tarayıcıda açar.

Çalıştırma:
  python view_results.py
"""

import os
import glob
import webbrowser
import tempfile
import pandas as pd

RESULTS_DIR = "results"


def _latest_run_number() -> int:
    existing = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    nums = set()
    for path in existing:
        base = os.path.basename(path)
        part = base.split("-", 1)[0]
        if part.isdigit():
            nums.add(int(part))
    return max(nums) if nums else None


def _df_to_html_table(df: pd.DataFrame) -> str:
    rows_html = ""
    for idx, row in df.iterrows():
        rows_html += "<tr>"
        rows_html += f"<td class='rank'>{idx}</td>"
        rows_html += f"<td class='word'>{row['Kelime']}</td>"
        rows_html += f"<td class='num'>{int(row['Frekans']):,}</td>"
        rows_html += f"<td class='score'>{row['Gini_Score']:.6f}</td>"
        rows_html += f"<td class='score'>{row['DFS_Score']:.6f}</td>"
        rows_html += "</tr>\n"
    return rows_html


def generate_html(run_no: int) -> str:
    eng_path = os.path.join(RESULTS_DIR, f"{run_no}-english_analysis.csv")
    tur_path = os.path.join(RESULTS_DIR, f"{run_no}-turkish_analysis.csv")

    df_eng = pd.read_csv(eng_path, encoding="utf-8-sig", index_col=0)
    df_tur = pd.read_csv(tur_path, encoding="utf-8-sig", index_col=0)
    df_eng.index.name = "Sıra"
    df_tur.index.name = "Sıra"

    eng_rows = _df_to_html_table(df_eng)
    tur_rows = _df_to_html_table(df_tur)

    total_eng = int(df_eng["Frekans"].sum())
    total_tur = int(df_tur["Frekans"].sum())

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gini & DFS Analizi – Çalıştırma #{run_no}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 32px 24px;
  }}

  header {{
    text-align: center;
    margin-bottom: 40px;
  }}

  header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }}

  header p {{
    color: #64748b;
    font-size: 0.9rem;
  }}

  .tabs {{
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-bottom: 32px;
  }}

  .tab-btn {{
    padding: 10px 28px;
    border: none;
    border-radius: 999px;
    font-family: inherit;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all .2s;
    background: #1e2130;
    color: #64748b;
  }}

  .tab-btn.active {{
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    color: #fff;
    box-shadow: 0 4px 15px rgba(99,102,241,.4);
  }}

  .panel {{ display: none; }}
  .panel.active {{ display: block; animation: fadeIn .3s ease; }}

  @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(8px) }} to {{ opacity:1; transform:none }} }}

  .stats-bar {{
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }}

  .stat-card {{
    flex: 1;
    min-width: 140px;
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 16px 20px;
  }}

  .stat-card .label {{
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 4px;
  }}

  .stat-card .value {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
  }}

  .search-box {{
    margin-bottom: 16px;
  }}

  .search-box input {{
    width: 100%;
    max-width: 360px;
    padding: 10px 16px;
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 8px;
    color: #e2e8f0;
    font-family: inherit;
    font-size: 0.9rem;
    outline: none;
    transition: border-color .2s;
  }}

  .search-box input:focus {{ border-color: #6366f1; }}
  .search-box input::placeholder {{ color: #475569; }}

  .table-wrap {{
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid #2d3148;
    max-height: 600px;
    overflow-y: auto;
  }}

  .table-wrap::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  .table-wrap::-webkit-scrollbar-track {{ background: #1e2130; }}
  .table-wrap::-webkit-scrollbar-thumb {{ background: #374151; border-radius: 3px; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }}

  thead th {{
    background: #181b2e;
    padding: 14px 18px;
    text-align: left;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #64748b;
    position: sticky;
    top: 0;
    z-index: 1;
    border-bottom: 1px solid #2d3148;
    white-space: nowrap;
  }}

  tbody tr {{
    border-bottom: 1px solid #1e2130;
    transition: background .15s;
  }}

  tbody tr:hover {{ background: #1e2540; }}
  tbody tr:last-child {{ border-bottom: none; }}

  td {{ padding: 11px 18px; vertical-align: middle; }}

  td.rank {{
    color: #475569;
    font-size: 0.8rem;
    font-weight: 500;
    width: 60px;
    text-align: center;
  }}

  td.word {{
    font-weight: 600;
    color: #a5b4fc;
    font-size: 0.9rem;
  }}

  td.num {{
    color: #34d399;
    font-weight: 500;
    text-align: right;
  }}

  td.score {{
    color: #e2e8f0;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-size: 0.82rem;
    color: #94a3b8;
  }}

  .gini-bar {{
    display: inline-block;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    margin-left: 8px;
    vertical-align: middle;
    min-width: 2px;
    transition: width .3s;
  }}

  .highlight td.word {{ color: #fbbf24; }}
</style>
</head>
<body>

<header>
  <h1>📊 Gini İndeksi & DFS Analizi</h1>
  <p>Çalıştırma #{run_no} · En sık kullanılan 500 kelime</p>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('eng', this)">🇬🇧 İngilizce SMS</button>
  <button class="tab-btn" onclick="switchTab('tur', this)">🇹🇷 Türkçe SMS</button>
</div>

<!-- ── İNGİLİZCE PANEL ── -->
<div id="panel-eng" class="panel active">
  <div class="stats-bar">
    <div class="stat-card"><div class="label">Analiz Edilen Kelime</div><div class="value">{len(df_eng)}</div></div>
    <div class="stat-card"><div class="label">Toplam Token</div><div class="value">{total_eng:,}</div></div>
    <div class="stat-card"><div class="label">Max Gini</div><div class="value">{df_eng['Gini_Score'].max():.4f}</div></div>
    <div class="stat-card"><div class="label">Max DFS</div><div class="value">{df_eng['DFS_Score'].max():.4f}</div></div>
  </div>
  <div class="search-box">
    <input type="text" id="search-eng" placeholder="Kelime ara…" oninput="filterTable('eng', this.value)">
  </div>
  <div class="table-wrap">
    <table id="table-eng">
      <thead>
        <tr>
          <th style="text-align:center">Sıra</th>
          <th>Kelime</th>
          <th style="text-align:right">Frekans</th>
          <th style="text-align:right">Gini Score ▼</th>
          <th style="text-align:right">DFS Score</th>
        </tr>
      </thead>
      <tbody>
{eng_rows}
      </tbody>
    </table>
  </div>
</div>

<!-- ── TÜRKÇE PANEL ── -->
<div id="panel-tur" class="panel">
  <div class="stats-bar">
    <div class="stat-card"><div class="label">Analiz Edilen Kelime</div><div class="value">{len(df_tur)}</div></div>
    <div class="stat-card"><div class="label">Toplam Token</div><div class="value">{total_tur:,}</div></div>
    <div class="stat-card"><div class="label">Max Gini</div><div class="value">{df_tur['Gini_Score'].max():.4f}</div></div>
    <div class="stat-card"><div class="label">Max DFS</div><div class="value">{df_tur['DFS_Score'].max():.4f}</div></div>
  </div>
  <div class="search-box">
    <input type="text" id="search-tur" placeholder="Kelime ara…" oninput="filterTable('tur', this.value)">
  </div>
  <div class="table-wrap">
    <table id="table-tur">
      <thead>
        <tr>
          <th style="text-align:center">Sıra</th>
          <th>Kelime</th>
          <th style="text-align:right">Frekans</th>
          <th style="text-align:right">Gini Score ▼</th>
          <th style="text-align:right">DFS Score</th>
        </tr>
      </thead>
      <tbody>
{tur_rows}
      </tbody>
    </table>
  </div>
</div>

<script>
function switchTab(lang, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + lang).classList.add('active');
  btn.classList.add('active');
}}

function filterTable(lang, query) {{
  const q = query.toLowerCase().trim();
  const rows = document.querySelectorAll('#table-' + lang + ' tbody tr');
  rows.forEach(row => {{
    const word = row.querySelector('td.word').textContent.toLowerCase();
    const show = !q || word.includes(q);
    row.style.display = show ? '' : 'none';
    row.classList.toggle('highlight', show && q.length > 0);
  }});
}}
</script>

</body>
</html>"""
    return html


def main():
    run_no = _latest_run_number()
    if run_no is None:
        print("results/ klasöründe hiç CSV bulunamadı. Önce 'python main.py' çalıştırın.")
        return

    print(f"Çalıştırma #{run_no} sonuçları yükleniyor…")
    html_content = generate_html(run_no)

    # Geçici HTML dosyası oluştur ve tarayıcıda aç
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8", prefix="gini_dfs_"
    )
    tmp.write(html_content)
    tmp.close()

    print(f"HTML oluşturuldu → {tmp.name}")
    webbrowser.open(f"file:///{tmp.name}")
    print("Tarayıcı açılıyor…")


if __name__ == "__main__":
    main()
