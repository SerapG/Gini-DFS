"""
data_loader.py
──────────────
Veri okuma ve metin ön işleme modülü.

Dışa açılan fonksiyonlar:
  load_english()  → İngilizce SMSSpamCollection'ı okur.
  load_turkish()  → TurkishSMS/ alt ağacını os.walk ile tarar.

Her iki fonksiyon da şu sütunları içeren bir pd.DataFrame döndürür:
  'label'  : 'spam' veya 'ham'
  'tokens' : preprocess() sonucu kelime listesi
"""

import os
import string
import pandas as pd

# ─── YAPILANDIRMA ──────────────────────────────────────────────────────────────

ENGLISH_FILE = "English_sms_spam/SMSSpamCollection"
TURKISH_DIR  = "TurkishSMS"

# Türkçe dosyalar için deneyecek kodlama sırası
_ENCODINGS   = ["utf-8", "latin-1", "cp1254", "iso-8859-9"]

# ASCII noktalama tablosu (Türkçe karakterler korunur: ş ı ğ ü ö ç İ)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ─── METİN ÖN İŞLEME ──────────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    """
    Metni küçük harfe çevirir, ASCII noktalamayı kaldırır ve kelimelere ayırır.

    Türkçe özeli: string.punctuation yalnızca ASCII karakterleri kapsadığından
    ş, ı, ğ, ü, ö, ç, İ harfleri bu adımda zarar görmez.

    Parametreler
    ------------
    text : str  Ham mesaj metni.

    Döndürür
    --------
    list[str]  Temizlenmiş token listesi.
    """
    text = str(text).lower()                 # küçük harfe çevir
    text = text.translate(_PUNCT_TABLE)      # ASCII noktalama kaldır
    return [t for t in text.split() if t]   # tokenize, boşları at


# ─── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────

def _safe_read(filepath: str) -> str:
    """
    Kodlama listesini sırayla deneyerek dosyayı okur.
    Tüm denemeler başarısız olursa UTF-8 + hata yoksayma ile okur.
    """
    for enc in _ENCODINGS:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _label_from_name(name: str) -> str | None:
    """
    Dosya ya da klasör adından etiket çıkarır.

    Kural:
      'spam'                  → 'spam'
      'ham' veya 'legitimate' → 'ham'
      hiçbiri                 → None (bu kaydı atla)
    """
    name = name.lower()
    if "spam" in name:
        return "spam"
    if "ham" in name or "legitimate" in name:
        return "ham"
    return None


# ─── VERİ OKUMA FONKSİYONLARI ─────────────────────────────────────────────────

def load_english() -> pd.DataFrame:
    """
    İngilizce SMSSpamCollection dosyasını tab-ayrımlı olarak okur.

    Dosya biçimi (başlık satırı yok):
        ham\\tMesaj metni
        spam\\tMesaj metni

    Döndürür
    --------
    pd.DataFrame  'label' ve 'tokens' sütunları.
    """
    df = pd.read_csv(
        ENGLISH_FILE,
        sep          = "\t",
        header       = None,
        names        = ["label", "text"],
        encoding     = "utf-8",
        on_bad_lines = "skip",
    )
    df["label"]  = df["label"].str.strip().str.lower()
    df["tokens"] = df["text"].apply(preprocess)
    return df[["label", "tokens"]]


def load_turkish() -> pd.DataFrame:
    """
    TurkishSMS/ klasörünü os.walk ile tarar, tüm .txt dosyalarını okur.

    Etiketleme: dosya adında 'spam' → spam, 'ham'/'legitimate' → ham.

    Dosya türleri:
      • Tek mesajlı  (≤ 5 satır): tüm içerik = 1 SMS
      • Çok satırlı  (> 5 satır): her satır  = 1 SMS

    Döndürür
    --------
    pd.DataFrame  'label' ve 'tokens' sütunları.
    """
    records = []

    for dirpath, _, filenames in os.walk(TURKISH_DIR):
        dir_label = _label_from_name(os.path.basename(dirpath))

        for filename in filenames:
            if not filename.lower().endswith(".txt"):
                continue

            # Önce dosya adından, bulamazsa klasör adından etiket al
            label = _label_from_name(filename) or dir_label
            if label is None:
                continue

            content = _safe_read(os.path.join(dirpath, filename)).strip()
            if not content:
                continue

            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

            if len(lines) <= 5:
                # Tek mesajlı dosya
                records.append({"label": label, "tokens": preprocess(content)})
            else:
                # Çok satırlı dosya: her satır ayrı SMS
                for line in lines:
                    records.append({"label": label, "tokens": preprocess(line)})

    return pd.DataFrame(records)
