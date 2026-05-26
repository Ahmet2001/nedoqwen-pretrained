"""Acronym / abbreviation expansion dictionary.

Maps uppercase acronyms to their Turkish expansions.  Used for:
- Acronym detection (is an ALL CAPS word a known acronym?)
- Expansion metadata (``_expansion`` field on ACRONYM tokens)
"""

from __future__ import annotations

ACRONYM_EXPANSIONS: dict[str, str] = {
    # ── International organizations ──────────────────────────────────────
    "NATO":     "Kuzey Atlantik Antlaşması Örgütü",
    "UN":       "Birleşmiş Milletler",
    "UNESCO":   "BM Eğitim, Bilim ve Kültür Örgütü",
    "UNICEF":   "BM Çocuklara Yardım Fonu",
    "WHO":      "Dünya Sağlık Örgütü",
    "IMF":      "Uluslararası Para Fonu",
    "WTO":      "Dünya Ticaret Örgütü",
    "EU":       "Avrupa Birliği",
    "INTERPOL": "Uluslararası Kriminal Polis Örgütü",
    "FIFA":     "Uluslararası Futbol Federasyonları Birliği",
    "IOC":      "Uluslararası Olimpiyat Komitesi",
    "UEFA":     "Avrupa Futbol Birliği",
    # ── Turkish institutions ─────────────────────────────────────────────
    "TBMM":    "Türkiye Büyük Millet Meclisi",
    "MEB":     "Milli Eğitim Bakanlığı",
    "TDK":     "Türk Dil Kurumu",
    "TTK":     "Türk Tarih Kurumu",
    "TCMB":    "Türkiye Cumhuriyet Merkez Bankası",
    "BDDK":    "Bankacılık Düzenleme ve Denetleme Kurumu",
    "SPK":     "Sermaye Piyasası Kurulu",
    "SGK":     "Sosyal Güvenlik Kurumu",
    "KDV":     "Katma Değer Vergisi",
    "ÖTV":     "Özel Tüketim Vergisi",
    "ÖSYM":    "Ölçme, Seçme ve Yerleştirme Merkezi",
    "YÖK":     "Yükseköğretim Kurulu",
    "TÜİK":    "Türkiye İstatistik Kurumu",
    "TÜBİTAK": "Türkiye Bilimsel ve Teknolojik Araştırma Kurumu",
    "ASELSAN":  "Askeri Elektronik Sanayii",
    "ODTÜ":     "Orta Doğu Teknik Üniversitesi",
    "İTÜ":      "İstanbul Teknik Üniversitesi",
    "METU":     "Middle East Technical University",
    "BOUN":     "Boğaziçi Üniversitesi",
    "İÜ":       "İstanbul Üniversitesi",
    "AU":       "Ankara Üniversitesi",
    "BİLKENT":  "İhsan Doğramacı Bilkent Üniversitesi",
    # ── Turkish exams ────────────────────────────────────────────────────
    "TUS":  "Tıpta Uzmanlık Sınavı",
    "DUS":  "Diş Hekimliğinde Uzmanlık Sınavı",
    "YDUS": "Yabancı Dil Uzmanlık Sınavı",
    "KPSS": "Kamu Personeli Seçme Sınavı",
    # ── Medical ──────────────────────────────────────────────────────────
    "CMV": "Sitomegalovirüs",  "EBV": "Epstein-Barr Virüsü",
    "VZV": "Varisella-Zoster Virüsü", "HHV": "İnsan Herpes Virüsü",
    "HSV": "Herpes Simplex Virüsü",   "HIV": "İnsan İmmün Yetmezlik Virüsü",
    "HBV": "Hepatit B Virüsü",        "HCV": "Hepatit C Virüsü",
    "RSV": "Respiratuar Sinsisyal Virüs", "HPV": "İnsan Papilloma Virüsü",
    "HAV": "Hepatit A Virüsü",
    "SLE": "Sistemik Lupus Eritematozus",
    "COPD": "Kronik Obstrüktif Akciğer Hastalığı",
    "DM":  "Diabetes Mellitus", "HTN": "Hipertansiyon",
    "MI":  "Miyokard İnfarktüsü", "DVT": "Derin Ven Trombozu",
    "PE":  "Pulmoner Emboli",
    "AML": "Akut Myeloid Lösemi",  "CML": "Kronik Myeloid Lösemi",
    "ALL": "Akut Lenfoblastik Lösemi", "CLL": "Kronik Lenfositik Lösemi",
    "ECG": "Elektrokardiyogram",   "EEG": "Elektroensefalogram",
    "MRI": "Manyetik Rezonans Görüntüleme",
    "CT":  "Bilgisayarlı Tomografi", "USG": "Ultrasonografi",
    "CBC": "Tam Kan Sayımı",
    "INR": "Uluslararası Normalleştirilmiş Oran",
    "LDL": "Düşük Yoğunluklu Lipoprotein",
    "HDL": "Yüksek Yoğunluklu Lipoprotein",
    "SMMM": "Serbest Muhasebeci Mali Müşavir",
    "YMM":  "Yeminli Mali Müşavir",
    "SM":   "Serbest Muhasebeci",
    # ── Technology ───────────────────────────────────────────────────────
    "AI":   "Yapay Zeka",        "ML":  "Makine Öğrenmesi",
    "LLM":  "Büyük Dil Modeli",  "NLP": "Doğal Dil İşleme",
    "API":  "Uygulama Programlama Arayüzü",
    "CPU":  "Merkezi İşlem Birimi", "GPU": "Grafik İşlem Birimi",
    "RAM":  "Rastgele Erişim Belleği",
    "SQL":  "Yapılandırılmış Sorgu Dili",
    "HTML": "HiperMetin İşaretleme Dili",
    "CSS":  "Basamaklı Stil Sayfaları",
    "OS":   "İşletim Sistemi",
    "BERT": "Çift Yönlü Kodlayıcı Temsiller",
    "GPT":  "Üretici Önceden Eğitilmiş Dönüştürücü",
    # ── Economics ────────────────────────────────────────────────────────
    "OPEC": "Petrol İhraç Eden Ülkeler Örgütü",
    "NAFTA": "Kuzey Amerika Serbest Ticaret Anlaşması",
    # ── Sports ───────────────────────────────────────────────────────────
    "NBA": "Ulusal Basketbol Birliği",
    "NFL": "Ulusal Futbol Ligi",
}
