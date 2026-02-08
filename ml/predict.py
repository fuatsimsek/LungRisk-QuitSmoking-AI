from pathlib import Path
import joblib
import pandas as pd

# Model dosya yolu
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lung_cancer_model.pkl"

# Eğitimde kullandığın feature kolonları
FEATURE_COLUMNS = [
    "age",
    "gender",
    "air_pollution",
    "dust_allergy",
    "occupational_hazards",
    "genetic_risk",
    "wheezing",
    "fatigue",
    "alcohol_use",
    "chronic_lung_disease",
    "smoking",
    "passive_smoker",
]

RISK_LABELS = {
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
}

RISK_LABELS_TR = {
    1: "DÜŞÜK",
    2: "ORTA",
    3: "YÜKSEK",
}

def ask_int(prompt: str, min_val: int, max_val: int) -> int:
    """
    Kullanıcıdan [min_val, max_val] aralığında tamsayı alır.
    Hatalı girişte tekrar sorar.
    """
    while True:
        try:
            value = int(input(f"{prompt} ({min_val}-{max_val}): "))
            if min_val <= value <= max_val:
                return value
            print(f"Lütfen {min_val}-{max_val} aralığında bir değer girin.")
        except ValueError:
            print("Lütfen sayısal bir değer girin.")


def get_risk_message(level: int) -> str:
    """
    Modelin tahmin ettiği risk seviyesine göre açıklama döner.
    Tıbbi tanı DEĞİL, sadece risk skoru olduğuna özellikle vurgu yapıyoruz.
    """
    if level == 1:
        return (
            "Verdiğiniz bilgilere göre model, akciğer kanseri açısından RİSKİNİZİ "
            "düşük seviyede değerlendiriyor. Bu sonuç bir TANI değildir, sadece "
            "risk skoru üretir. Sigara kullanıyorsanız bırakmanız, düzenli "
            "kontroller yaptırmanız yine de önemlidir."
        )
    elif level == 2:
        return (
            "Verdiğiniz bilgilere göre model, akciğer kanseri açısından RİSKİNİZİ "
            "orta seviyede değerlendiriyor. Özellikle sigara, pasif sigara maruziyeti "
            "ve genetik risk faktörleri önemlidir. En kısa sürede bir göğüs "
            "hastalıkları uzmanına veya aile hekiminize başvurup ayrıntılı değerlendirme "
            "yaptırmanız faydalı olacaktır. Bu sistem kesin tanı koymaz; sadece erken "
            "uyarı ve farkındalık amaçlıdır."
        )
    else:  # level == 3
        return (
            "Verdiğiniz bilgilere göre model, akciğer kanseri açısından RİSKİNİZİ "
            "YÜKSEK seviyede değerlendiriyor. Sigara, pasif içicilik, kronik akciğer "
            "hastalığı veya çevresel riskler (hava kirliliği, mesleki maruziyet) "
            "önemli rol oynuyor olabilir. Bu sonuç acil tıbbi değerlendirme ihtiyacına "
            "işaret edebilir. En kısa sürede bir sağlık kuruluşuna başvurmanız ve "
            "doktor kontrolü ile gerekli tetkikleri yaptırmanız önemlidir. Bu sistem "
            "bir tanı aracı değil, sadece risk skoru üreten yardımcı bir modeldir."
        )


def build_html_output(
    level: int, probs, input_data: dict
) -> str:
    """
    HTML formatında özet çıktı üretir.
    Bunu Django template içine gömebilirsin.
    """
    low_p, med_p, high_p = probs
    risk_en = RISK_LABELS.get(level, "UNKNOWN")
    risk_tr = RISK_LABELS_TR.get(level, "BİLİNMİYOR")
    message = get_risk_message(level)

    # Basit, inline stilli bir HTML kartı
    color_map = {
        1: "#22c55e",  # green
        2: "#f97316",  # orange
        3: "#ef4444",  # red
    }
    badge_color = color_map.get(level, "#6b7280")

    html = f"""
<div style="border:1px solid #e5e7eb; border-radius:12px; padding:16px; max-width:520px; font-family:system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <h2 style="margin-top:0; margin-bottom:8px; font-size:20px;">Akciğer Kanseri Risk Değerlendirmesi</h2>
  <span style="display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; color:white; background:{badge_color};">
    Risk Seviyesi: {risk_tr} ({risk_en})
  </span>
  <div style="margin-top:12px; font-size:14px; line-height:1.6;">
    <p style="margin:0 0 8px 0;"><strong>Model değerlendirmesi:</strong> {message}</p>
    <p style="margin:0 0 4px 0;"><strong>Risk Olasılıkları:</strong></p>
    <ul style="margin:4px 0 8px 18px; padding:0; font-size:13px;">
      <li>LOW (Düşük)   : %{low_p:.1f}</li>
      <li>MEDIUM (Orta) : %{med_p:.1f}</li>
      <li>HIGH (Yüksek) : %{high_p:.1f}</li>
    </ul>
    <p style="margin:0; font-size:12px; color:#6b7280;">
      Bu çıktı yalnızca eğitim amaçlı bir risk skorudur; tıbbi tanı yerine geçmez.
      Herhangi bir şüpheniz varsa mutlaka bir doktora başvurun.
    </p>
  </div>
</div>
"""
    return html


def main():
    print("🔬 Akciğer Kanseri Risk Değerlendirme Sihirbazı\n")

    # Modeli yükle
    model_data = joblib.load(MODEL_PATH)

    # Eğer dict olarak kaydedildiyse:
    if isinstance(model_data, dict):
        model = model_data["model"]
    else:
        model = model_data

    # Kullanıcıdan ölçek tabanlı sorular
    answers = {}

    print("Lütfen aşağıdaki soruları dürüstçe yanıtlayın. Ölçekler 1-8 / 1-9 arasıdır.\n")

    answers["age"] = ask_int("Yaşınız", 14, 73)

    print("\nCinsiyetiniz:")
    print("1: Erkek")
    print("2: Kadın")
    answers["gender"] = ask_int("Seçiminiz", 1, 2)

    print("\nYaşadığınız ortamın hava kirliliği seviyesi:")
    print("1: Çok düşük  ...  8: Çok yüksek")
    answers["air_pollution"] = ask_int("Hava kirliliği düzeyi", 1, 8)

    print("\nToz alerjisi (alerjik hassasiyet):")
    print("1: Yok / Çok hafif  ...  8: Çok şiddetli")
    answers["dust_allergy"] = ask_int("Toz alerjisi düzeyi", 1, 8)

    print("\nMesleki risk (kimyasal, toz, duman vb. maruziyet):")
    print("1: Yok  ...  8: Çok yüksek riskli ortam")
    answers["occupational_hazards"] = ask_int("Mesleki risk düzeyi", 1, 8)

    print("\nGenetik risk (ailede akciğer veya ciddi kanser öyküsü):")
    print("1: Yok  ...  7: Çok güçlü aile öyküsü")
    answers["genetic_risk"] = ask_int("Genetik risk düzeyi", 1, 7)

    print("\nHırıltılı solunum (wheezing) sıklığı:")
    print("1: Hiç / Çok nadir  ...  8: Çok sık")
    answers["wheezing"] = ask_int("Hırıltı sıklığı", 1, 8)

    print("\nYorgunluk (fatigue) seviyesi:")
    print("1: Neredeyse hiç  ...  9: Günlük yaşamı ciddi etkiliyor")
    answers["fatigue"] = ask_int("Yorgunluk düzeyi", 1, 9)

    print("\nAlkol kullanımı düzeyi:")
    print("1: Hiç / Çok nadir  ...  8: Çok sık / Yüksek miktarda")
    answers["alcohol_use"] = ask_int("Alkol kullanımı düzeyi", 1, 8)

    print("\nKronik akciğer hastalığı (örn. KOAH, astım) durumu:")
    print("1: Yok / Hafif  ...  7: Ağır / Uzun süreli hastalık")
    answers["chronic_lung_disease"] = ask_int(
        "Kronik akciğer hastalığı düzeyi", 1, 7
    )

    print("\nSigara kullanımı (aktif içicilik):")
    print("1: Hiç  ...  8: Çok yoğun içici")
    answers["smoking"] = ask_int("Sigara kullanımı düzeyi", 1, 8)

    print("\nPasif içicilik (yanınızdaki kişilerin sigara dumanına maruziyet):")
    print("1: Hemen hemen hiç  ...  8: Sürekli maruziyet")
    answers["passive_smoker"] = ask_int("Pasif içicilik düzeyi", 1, 8)

    # DataFrame oluştur
    df = pd.DataFrame([answers], columns=FEATURE_COLUMNS)

    # Tahmin
    pred_class = model.predict(df)[0]
    proba = model.predict_proba(df)[0]  # [p_low, p_medium, p_high]

    risk_en = RISK_LABELS.get(pred_class, "UNKNOWN")
    risk_tr = RISK_LABELS_TR.get(pred_class, "BİLİNMİYOR")
    msg = get_risk_message(pred_class)

    print("\n================ SONUÇ ================\n")
    print(f"Tahmini Risk Seviyeniz: {risk_tr} ({risk_en})\n")
    print("Detaylı Açıklama:")
    print(msg)
    print("\nRisk Olasılıkları:")
    print(f"  LOW (Düşük)   : %{proba[0]*100:.1f}")
    print(f"  MEDIUM (Orta) : %{proba[1]*100:.1f}")
    print(f"  HIGH (Yüksek) : %{proba[2]*100:.1f}")
    print(
        "\nNOT: Bu çıktı sadece eğitim amaçlı bir risk değerlendirmesidir; "
        "tıbbi tanı yerine geçmez. Şüpheniz varsa mutlaka doktora başvurun."
    )

    # HTML çıktısı
    html_output = build_html_output(pred_class, proba * 100, answers)
    print("\n=============== HTML ÇIKTISI (Kopyalayıp frontend'de kullanabilirsin) ===============\n")
    print(html_output)


if __name__ == "__main__":
    main()
