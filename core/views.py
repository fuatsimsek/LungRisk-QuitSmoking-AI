import os
import joblib
from pathlib import Path
import pandas as pd

from django.shortcuts import render, redirect
from django.utils import timezone
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import QuitPrediction, CancerRiskRecord
from .serializers import CancerRiskRecordSerializer


MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "lung_cancer_model.pkl"
# Daha iyi kalibre edilmiş sigara bırakma modeli
QUIT_MODEL_PATH = Path(__file__).resolve().parent.parent / "ml2" / "model_quityrs_rf_calibrated.pkl"
DECISION_THRESHOLD = float(os.getenv("QUIT_DECISION_THRESHOLD", "0.6"))
AGE_MEAN = 50.0
AGE_STD = 12.0


def _minmax(value: int, target_max: int = 10) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 1.0
    v = max(1.0, min(target_max, v))
    return (v - 1.0) / (target_max - 1.0)


def _build_suggestions(raw: dict, level: str) -> list[str]:
    tips: list[str] = []
    lvl = (level or "").upper()

    # General message by risk level
    if lvl == "HIGH":
        tips.append("High risk detected. Urgent medical evaluation is recommended.")
    elif lvl == "MEDIUM":
        tips.append("Medium risk present. See a specialist as soon as possible.")
    else:
        tips.append("Risk is low. One check-up per year is enough; keep living healthily.")

    def hi(key: str, threshold: float) -> bool:
        try:
            return float(raw.get(key, 0)) >= threshold
        except Exception:
            return False

    if hi("smoking", 7):
        tips.append("Smoking: Cut by 50%, aim for under 10 cigarettes per day.")
    elif hi("smoking", 5):
        tips.append("Smoking: Reduce; consider nicotine support or counselling.")

    if hi("alcohol_use", 7):
        tips.append("Alcohol: Limit to 1–2 units per week; avoid binge drinking.")
    elif hi("alcohol_use", 5):
        tips.append("Alcohol: Gradually reduce intake to under 3–4 units per week.")

    if hi("air_pollution", 6) or hi("passive_smoker", 6):
        tips.append("Air quality/passive smoking: Use a mask, seek clean air; avoid smoky environments.")

    if hi("chronic_lung_disease", 6):
        tips.append("Chronic lung disease: Increase respiratory check-ups; keep up medication compliance.")

    if hi("fatigue", 7):
        tips.append("Fatigue: Review sleep and activity; consider a check-up if needed.")

    # Top risk factor–specific advice
    factor_keys = [
        "smoking",
        "passive_smoker",
        "air_pollution",
        "dust_allergy",
        "occupational_hazards",
        "genetic_risk",
        "wheezing",
        "fatigue",
        "alcohol_use",
        "chronic_lung_disease",
    ]
    max_key = None
    max_val = -1
    for k in factor_keys:
        try:
            v = float(raw.get(k, 0))
        except Exception:
            v = 0
        if v > max_val:
            max_val = v
            max_key = k

    max_tip_map = {
        "smoking": "Top factor: smoking. Reduce by 60%; consider professional support and nicotine replacement.",
        "passive_smoker": "Top factor: passive smoking. Strictly limit smoky environments; improve ventilation and use a mask indoors.",
        "air_pollution": "Top factor: air pollution. Use a mask/filter when pollution is high; prefer an air purifier indoors.",
        "dust_allergy": "Top factor: dust/allergy. Regular cleaning, HEPA filter; see a doctor about antihistamines.",
        "occupational_hazards": "Top factor: occupational exposure. Use PPE and get occupational safety advice.",
        "genetic_risk": "Top factor: genetic risk. Plan periodic screening and specialist follow-up due to family history.",
        "wheezing": "Top factor: wheezing. Lung function tests and respiratory check-up are recommended.",
        "fatigue": "Top factor: fatigue. Improve sleep, stress and diet; consider an internal medicine check-up.",
        "alcohol_use": "Top factor: alcohol. Minimise weekly intake; consider support programmes.",
        "chronic_lung_disease": "Top factor: chronic lung disease. Regular medication compliance and close specialist follow-up are needed.",
    }
    if max_key and max_key in max_tip_map:
        tips.append(max_tip_map[max_key])

    return tips


# ============================
# QUIT MODEL HELPERS
# ============================
def _apply_imputers(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    df = df.copy()
    for col, val in imputers.get("mode", {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in imputers.get("median", {}).items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


def _apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            ser = df[col]
            try:
                ser = ser.astype(enc.classes_.dtype)
            except Exception:
                ser = ser.astype(str)
            mask_unknown = ~ser.isin(enc.classes_)
            if mask_unknown.any():
                ser.loc[mask_unknown] = enc.classes_[0]
            df[col] = enc.transform(ser)
    return df


def _quit_advice(prob: float) -> str:
    if prob < 0.33:
        return (
            "Your quit likelihood appears low. Reducing daily use, "
            "support groups or nicotine replacement may make the process easier."
        )
    elif prob < 0.66:
        return (
            "You have moderate quit potential. Your attempts are positive but the habit is strong. "
            "Adding small changes to your routine can boost motivation."
        )
    else:
        return (
            "Your quit likelihood is quite high. "
            "Making a detailed plan and reducing triggers can increase your chance of success."
        )


# ============================
# TAHMİN API (AKCİĞER)
# ============================

@method_decorator(csrf_exempt, name="dispatch")
class LungRiskPredictAPIView(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        return Response(
            {"error": "This endpoint is only called via POST from the form."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def post(self, request):
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"] if isinstance(model_data, dict) else model_data

        data = request.data

        raw = {
            "age": int(data.get("age", 0)),
            "gender": int(data.get("gender", 1)),  # 1=Male, 2=Female
            "air_pollution": int(data.get("air_pollution", 1)),
            "dust_allergy": int(data.get("dust_allergy", 1)),
            "occupational_hazards": int(data.get("occupational_hazards", 1)),
            "genetic_risk": int(data.get("genetic_risk", 1)),
            "wheezing": int(data.get("wheezing", 1)),
            "fatigue": int(data.get("fatigue", 1)),
            "alcohol_use": int(data.get("alcohol_use", 1)),
            "chronic_lung_disease": int(data.get("chronic_lung_disease", 1)),
            "smoking": int(data.get("smoking", 1)),
            "passive_smoker": int(data.get("passive_smoker", 1)),
        }

        df = pd.DataFrame([raw])

        risk_idx = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0]

        risk_map = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
        risk_level = risk_map.get(risk_idx, "LOW")

        suggestions = _build_suggestions(raw, risk_level)

        user_obj = None
        user_id = request.session.get("user_id")
        if user_id:
            user_obj = User.objects.filter(id=user_id).first()

        record = CancerRiskRecord.objects.create(
            user=user_obj,
            predicted_risk_level=risk_level,
            **raw,
            suggestions=suggestions,
            last_suggestion=suggestions[0] if suggestions else None,
            created_at=timezone.now(),
        )

        return Response(
            {
                "risk_level": risk_level,
                "probabilities": {
                    "low": float(proba[0]),
                    "medium": float(proba[1]),
                    "high": float(proba[2]),
                },
                "inputs": raw,
                "suggestions": suggestions,
                "record_id": record.id,
            }
        )


# ============================
# GEÇMİŞ (AKCİĞER)
# ============================

class LungRiskHistoryAPIView(APIView):
    def get(self, request):
        user_obj = None
        user_id = request.session.get("user_id")
        if user_id:
            user_obj = User.objects.filter(id=user_id).first()

        qs = CancerRiskRecord.objects.order_by("-created_at")
        if user_obj:
            qs = qs.filter(user=user_obj)

        records = qs[:50].values(
            "id",
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
            "predicted_risk_level",
            "suggestions",
            "last_suggestion",
            "created_at",
        )
        enriched = []
        for rec in records:
            lvl = rec.get("predicted_risk_level") or "LOW"
            stored_sugs = rec.get("suggestions")
            if not stored_sugs:
                stored_sugs = _build_suggestions(rec, lvl)
            rec["suggestions"] = stored_sugs
            rec["last_suggestion"] = rec.get("last_suggestion") or (
                stored_sugs[0] if stored_sugs else None
            )
            enriched.append(rec)
        return Response(enriched)


# ============================
# QUIT SMOKING API
# ============================

@method_decorator(csrf_exempt, name="dispatch")
class QuitSmokingPredictAPIView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        user_obj = None
        user_id = request.session.get("user_id")
        if user_id:
            user_obj = User.objects.filter(id=user_id).first()

        artifact = joblib.load(QUIT_MODEL_PATH)
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        threshold = artifact.get("threshold", 0.8)
        imputers = artifact.get("imputers", {})
        encoders = artifact.get("encoders", {})

        data = request.data

        def map_to_nhis(v, mapping):
            try:
                iv = int(v)
            except Exception:
                return v
            return mapping.get(iv, iv)

        # Form girildiğinde küçük ölçekleri NHIS kodlarına çevir
        raw_in = {
            "AGE": data.get("AGE"),
            "GENDER": data.get("GENDER"),
            "EDUC": data.get("EDUC"),
            "ALCSTAT1": data.get("ALCSTAT1"),
            "ALCDAYSYR": data.get("ALCDAYSYR"),
            "CIGSDAY": data.get("CIGSDAY"),
            "CSQTRYYR": data.get("CSQTRYYR"),
            "CSWANTQUIT": data.get("CSWANTQUIT"),
            "DEPRX": data.get("DEPRX"),
            "DEPFEELEVL": data.get("DEPFEELEVL"),
            "CIGSLONGFS": data.get("CIGSLONGFS"),
            "QUITNO": data.get("QUITNO"),
        }

        # EDUC 1–5 ise NHIS kodu 114/201/301/400/500
        educ_map = {1: 114, 2: 201, 3: 301, 4: 400, 5: 500}
        # ALCSTAT1 1–4 ise NHIS 1/2/3/3 (4'ü ağır kabul edip 3'e çekiyoruz)
        alc_map = {1: 1, 2: 2, 3: 3, 4: 3}

        mapped = dict(raw_in)
        mapped["EDUC"] = map_to_nhis(raw_in.get("EDUC"), educ_map)
        mapped["ALCSTAT1"] = map_to_nhis(raw_in.get("ALCSTAT1"), alc_map)

        # Sayısal alanları güvenli int/float'a çek
        for key in ["AGE", "ALCDAYSYR", "CIGSDAY", "CSQTRYYR", "CSWANTQUIT", "DEPRX", "DEPFEELEVL", "GENDER"]:
            try:
                mapped[key] = float(mapped.get(key)) if mapped.get(key) is not None else None
            except Exception:
                pass

        raw = mapped

        df = pd.DataFrame([raw])
        df = _apply_imputers(df, imputers)
        df = _apply_encoders(df, encoders)

        # Eksik feature tamamlama
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Modelin fit sırasında beklediği sıra/dubleler olabilir; varsa onu kullan.
        model_feature_order = list(getattr(model, "feature_names_in_", feature_names))
        X = df.reindex(columns=model_feature_order, fill_value=0)

        # classes_ üzerinden güvenli probability
        proba_full = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            idx = classes.index(1)
        else:
            idx = 1 if len(classes) > 1 else 0
        proba = float(proba_full[idx])

        # Hafif iyimser ayar: bırakmak istiyor ve deneme yaptıysa küçük bonus
        try:
            want = float(raw.get("CSWANTQUIT") or 0)
        except Exception:
            want = 0
        try:
            tries = float(raw.get("CSQTRYYR") or 0)
        except Exception:
            tries = 0
        bonus = 1.0
        if want == 1:
            bonus *= 1.15
        if tries > 0:
            bonus *= 1.10
        proba = max(0.0, min(1.0, proba * bonus))

        # Binary karar + seviyeler
        decision_threshold = DECISION_THRESHOLD
        can_quit = proba >= decision_threshold
        if proba < 0.4:
            level = "low"
        elif proba < 0.9:
            level = "medium"
        else:
            level = "high"

        advice = _quit_advice(proba)

        # zorlayan faktörler / destekler
        factors = []
        obstacles = []
        try:
            cigs = float(raw.get("CIGSDAY") or 0)
            if cigs > 10:
                obstacles.append("Your daily cigarette count is high.")
            elif cigs <= 5:
                factors.append("Keeping daily cigarettes low is a big advantage.")
        except Exception:
            pass
        try:
            tries = float(raw.get("CSQTRYYR") or 0)
            if tries == 0:
                obstacles.append("No quit attempts in the last 12 months.")
            else:
                factors.append("Your quit attempts boost motivation.")
        except Exception:
            pass
        try:
            alc = float(raw.get("ALCDAYSYR") or 0)
            if alc > 100:
                obstacles.append("Your alcohol use is high.")
            elif alc <= 20:
                factors.append("Keeping alcohol limited raises quit likelihood.")
        except Exception:
            pass
        try:
            want = float(raw.get("CSWANTQUIT") or 0)
            if want == 1:
                factors.append("Wanting to quit is a major plus.")
        except Exception:
            pass
        try:
            longest = float(raw.get("CIGSLONGFS") or 0)
            if longest == 0:
                obstacles.append("You haven't quit for long before; create a step-by-step trial plan.")
            elif longest >= 30:
                factors.append(f"Having quit for {int(longest)} days before is a big advantage.")
        except Exception:
            pass
        try:
            qno = float(raw.get("QUITNO") or 0)
            if qno >= 5:
                factors.append("You have many quit attempts; you can use that experience.")
            elif qno == 0:
                obstacles.append("No total quit attempts; create a structured plan.")
        except Exception:
            pass

        if not factors and not obstacles:
            factors.append("No clear driving factor identified.")

        # Suggestion set by decision
        suggestions = []
        if can_quit:
            suggestions.extend(factors or [])
            suggestions.append("Set a clear quit date and list your triggers.")
            suggestions.append("Book an appointment with a health professional for nicotine replacement or counselling.")
            suggestions.append("Add small alternative tasks at trigger times (e.g. 10 min walk, drinking water).")
            suggestions.append("Make a follow-up pact with a friend/partner; give short daily feedback.")
            suggestions.append("Minimise alcohol during the quit period; avoid alcohol as a trigger in the first 2 weeks.")
        else:
            suggestions.extend(obstacles or [])
            suggestions.append("Make a 2-week step-down plan to cut daily cigarettes by 50% (e.g. 10→5).")
            suggestions.append("Keep a 1-week trial log: when and where you smoke and how you feel.")
            suggestions.append("Try nicotine gum/patch or similar support in a trial week before quitting.")
            suggestions.append("Add short exercise or breathing routine twice daily for sleep and stress.")
            suggestions.append("Cut alcohol sharply in the first 10 days to weaken the trigger.")

        # Kaydet
        QuitPrediction.objects.create(
            user=user_obj,
            probability=proba,
            risk_level=level,
            advice_text=advice,
            age=raw.get("AGE") or 0,
            gender=raw.get("GENDER") or 0,
            educ=raw.get("EDUC") or 0,
            alcstat1=raw.get("ALCSTAT1") or 0,
            alcdaysyr=raw.get("ALCDAYSYR") or 0,
            cigsday=raw.get("CIGSDAY") or 0,
            csqtryyr=raw.get("CSQTRYYR") or 0,
            cswantquit=raw.get("CSWANTQUIT") or 0,
            deprx=raw.get("DEPRX") or 0,
            deplevel=raw.get("DEPFEELEVL") or 0,
            suggestions=suggestions or (factors or obstacles),
            last_suggestion=(suggestions or factors or obstacles)[0] if (suggestions or factors or obstacles) else None,
        )

        return Response(
            {
                "probability": proba,
                "can_quit": can_quit,
                "risk_level": level,
                "decision_threshold_used": decision_threshold,
                "factors": factors,
                "obstacles": obstacles,
                "suggestions": suggestions or (factors or obstacles),
                "advice_text": advice,
            }
        )


class QuitSmokingHistoryAPIView(APIView):
    def get(self, request):
        user_id = request.session.get("user_id")
        qs = QuitPrediction.objects.order_by("-created_at")
        if user_id:
            qs = qs.filter(user_id=user_id)
        qs = qs[:50]
        out = []
        for r in qs:
            out.append(
                {
                    "created_at": r.created_at,
                    "probability": float(r.probability),
                    "can_quit": float(r.probability) >= DECISION_THRESHOLD,
                    # Frontend tabloda büyük harfli key'ler bekleniyor.
                    "CIGSDAY": r.cigsday,
                    "ALCDAYSYR": r.alcdaysyr,
                    "CSQTRYYR": r.csqtryyr,
                    "CSWANTQUIT": r.cswantquit,
                    "deplevel": r.deplevel,
                    "alcstat1": r.alcstat1,
                    "educ": r.educ,
                    "age": r.age,
                    "gender": r.gender,
                    "risk_level": r.risk_level,
                    "advice_text": r.advice_text,
                    "suggestions": r.suggestions,
                }
        )
        return Response(out)


# ============================
# SAYFALAR
# ============================

def login_page(request):
    if request.method == "POST":
        email = (request.POST.get("email") or "").strip().lower()
        password = request.POST.get("password") or ""
        user = User.objects.filter(email=email).first()
        if user and user.check_password(password):
            request.session["user_id"] = user.id
            return redirect("/dashboard/")
        return render(request, "login.html", {"error": "Invalid email or password"})
    return render(request, "login.html")


def register_page(request):
    if request.method == "POST":
        email = (request.POST.get("email") or "").strip().lower()
        password = request.POST.get("password") or ""
        password_confirm = request.POST.get("password_confirm") or ""
        if not email or not password:
            return render(request, "register.html", {"error": "Email and password are required"})
        if password != password_confirm:
            return render(request, "register.html", {"error": "Passwords do not match"})
        if User.objects.filter(email=email).exists():
            return render(request, "register.html", {"error": "This email is already registered"})
        user = User.objects.create_user(username=email, email=email, password=password)
        request.session["user_id"] = user.id
        return redirect("/dashboard/")
    return render(request, "register.html")


def logout_page(request):
    request.session.flush()
    return redirect("/login/")


def root_view(request):
    if request.session.get("user_id"):
        return redirect("/dashboard/")
    return redirect("/login/")


def risk_form_page(request):
    if not request.session.get("user_id"):
        return redirect("/login/")
    return render(request, "risk_form.html")


def dashboard_page(request):
    if not request.session.get("user_id"):
        return redirect("/login/")
    return render(request, "dashboard.html")


def quit_page(request):
    if not request.session.get("user_id"):
        return redirect("/login/")
    return render(request, "quit_dashboard.html")


def quit_form_page(request):
    if not request.session.get("user_id"):
        return redirect("/login/")
    return render(request, "quit_form.html")
