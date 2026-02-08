from django.db import models
from django.contrib.auth.models import User


# --------------------------
# Sigara Bırakma Kayıt Modeli
# --------------------------
class QuitPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    probability = models.FloatField()
    risk_level = models.CharField(max_length=20, default="medium")
    advice_text = models.TextField(blank=True, null=True)
    suggestions = models.JSONField(default=list, blank=True)
    last_suggestion = models.CharField(max_length=255, blank=True, null=True)

    age = models.IntegerField()
    gender = models.IntegerField()
    educ = models.IntegerField()
    alcstat1 = models.IntegerField()
    alcdaysyr = models.IntegerField()
    cigsday = models.IntegerField()
    csqtryyr = models.IntegerField()
    cswantquit = models.IntegerField()
    deprx = models.IntegerField()
    deplevel = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"QuitPrediction {self.user} - {self.probability:.3f}"


# --------------------------
# Kanser Risk Kayıt Modeli
# --------------------------
class CancerRiskRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    age = models.IntegerField()
    gender = models.IntegerField()

    air_pollution = models.IntegerField()
    dust_allergy = models.IntegerField()
    occupational_hazards = models.IntegerField()
    genetic_risk = models.IntegerField()
    wheezing = models.IntegerField()
    fatigue = models.IntegerField()
    alcohol_use = models.IntegerField()
    chronic_lung_disease = models.IntegerField()
    smoking = models.IntegerField()
    passive_smoker = models.IntegerField()

    predicted_risk_level = models.CharField(max_length=50)
    suggestions = models.JSONField(default=list, blank=True)
    last_suggestion = models.CharField(max_length=255, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"CancerRiskRecord {self.user} - {self.predicted_risk_level}"
