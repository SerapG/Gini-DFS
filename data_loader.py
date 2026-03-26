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
import warnings
import pandas as pd

# --- İSTEĞE BAĞLI KÜTÜPHANELER VE GLOBAL NESNELER ---
# Nesneleri globalde tanımlıyoruz ki her SMS için tekrar tekrar yüklenmesin (Performans için)
EN_STOPWORDS = set()
TR_STOPWORDS = set()
en_stemmer = None
tr_stemmer = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    EN_STOPWORDS = set(stopwords.words('english'))
    TR_STOPWORDS = set(stopwords.words('turkish'))
    en_stemmer = SnowballStemmer("english")
except ImportError:
    warnings.warn("NLTK kütüphanesi bulunamadı! Stopwords/Stemming kullanılamayacak. 'pip install nltk' ile yükleyebilirsiniz.")

try:
    from TurkishStemmer import TurkishStemmer
    tr_stemmer = TurkishStemmer()
except ImportError:
    warnings.warn("TurkishStemmer kütüphanesi bulunamadı! Türkçe için stemming yapılamayacak. 'pip install TurkishStemmer' kullanın.")

# ─── YAPILANDIRMA ──────────────────────────────────────────────────────────────

ENGLISH_FILE = "English_sms_spam/SMSSpamCollection"
TURKISH_DIR  = "TurkishSMS"

# Türkçe dosyalar için deneyecek kodlama sırası
_ENCODINGS   = ["utf-8", "latin-1", "cp1254", "iso-8859-9"]

# ASCII noktalama tablosu (Türkçe karakterler korunur: ş ı ğ ü ö ç İ)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ─── METİN ÖN İŞLEME ──────────────────────────────────────────────────────────

def preprocess_basic(text: str) -> list:
    """
    1. Fonksiyon: Yalnızca küçük harfe çevirme, noktalama temizleme ve tokenization yapar.
    """
    text = str(text).lower()                 # küçük harfe çevir
    text = text.translate(_PUNCT_TABLE)      # ASCII noktalama kaldır
    return [t for t in text.split() if t]    # tokenize, boşları at


def preprocess_with_stopwords(text: str, language: str = "english") -> list:
    """
    2. Fonksiyon: Temel ön işleme (küçük harf, tokenization) + Stop-word temizliği yapar.
    """
    tokens = preprocess_basic(text)
    if language == "english" and EN_STOPWORDS:
        return [t for t in tokens if t not in EN_STOPWORDS]
    elif language == "turkish" and TR_STOPWORDS:
        return [t for t in tokens if t not in TR_STOPWORDS]
    return tokens


def preprocess_with_stemming(text: str, language: str = "english") -> list:
    """
    3. Fonksiyon: Temel ön işleme (küçük harf, tokenization) + Stemming (Kök bulma) yapar.
    """
    tokens = preprocess_basic(text)
    if language == "english" and en_stemmer:
        return [en_stemmer.stem(t) for t in tokens]
    elif language == "turkish" and tr_stemmer:
        return [tr_stemmer.stem(t) for t in tokens]
    return tokens


def preprocess_full(text: str, language: str = "english") -> list:
    """
    4. Fonksiyon: Hepsi bir arada! (Temel + Stop-words + Stemming)
    """
    tokens = preprocess_with_stopwords(text, language)
    if language == "english" and en_stemmer:
        return [en_stemmer.stem(t) for t in tokens]
    elif language == "turkish" and tr_stemmer:
        return [tr_stemmer.stem(t) for t in tokens]
    return tokens


def preprocess(
    text: str,
    language: str = "english",
    remove_stopwords: bool = False,
    apply_stemming: bool = False
) -> list:
    """
    Yönlendirici Fonksiyon: Parametrelere bakarak yukarıdaki 4 ana fonksiyondan uygun olanını çağırır.
    Böylece data_loader.py içindeki diğer yapılar bozulmaz.
    """
    if remove_stopwords and apply_stemming:
        return preprocess_full(text, language)
    elif remove_stopwords:
        return preprocess_with_stopwords(text, language)
    elif apply_stemming:
        return preprocess_with_stemming(text, language)
    else:
        return preprocess_basic(text)


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

def load_english(remove_stopwords: bool = False, apply_stemming: bool = False) -> pd.DataFrame:
    """
    İngilizce SMSSpamCollection dosyasını tab-ayrımlı olarak okur.

    Dosya biçimi (başlık satırı yok):
        ham\tMesaj metni
        spam\tMesaj metni

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
    df["tokens"] = df["text"].apply(
        lambda t: preprocess(t, language="english", 
                             remove_stopwords=remove_stopwords, 
                             apply_stemming=apply_stemming)
    )
    return df[["label", "tokens"]]


def load_turkish(remove_stopwords: bool = False, apply_stemming: bool = False) -> pd.DataFrame:
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
                records.append({
                    "label": label, 
                    "tokens": preprocess(content, language="turkish", remove_stopwords=remove_stopwords, apply_stemming=apply_stemming)
                })
            else:
                # Çok satırlı dosya: her satır ayrı SMS
                for line in lines:
                    records.append({
                        "label": label, 
                        "tokens": preprocess(line, language="turkish", remove_stopwords=remove_stopwords, apply_stemming=apply_stemming)
                    })

    return pd.DataFrame(records)