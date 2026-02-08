from django.urls import path
from .views import (
    LungRiskHistoryAPIView,
    LungRiskPredictAPIView,
    QuitSmokingPredictAPIView,
    QuitSmokingHistoryAPIView,
    login_page,
    register_page,
    logout_page,
    risk_form_page,
    dashboard_page,
    quit_page,
    quit_form_page,
    root_view,
)

urlpatterns = [
    # === SAYFALAR ===
    path("", root_view, name="root"),

    path("login/", login_page, name="login-page"),
    path("register/", register_page, name="register-page"),
    path("logout/", logout_page, name="logout-page"),

    # Kanser riski hesaplama
    path("risk-form/", risk_form_page, name="risk-form-page"),
    path("dashboard/", dashboard_page, name="dashboard-page"),

    # Sigara bırakma olasılığı
    path("quit-form/", quit_form_page, name="quit-form-page"),
    path("dashboard2/", quit_page, name="dashboard2-page"),

    # === API ENDPOINTS ===
    path("api/risk-form/", LungRiskPredictAPIView.as_view(), name="api-risk-form"),
    path("api/history/", LungRiskHistoryAPIView.as_view(), name="api-history"),

    path("api/quit-smoking-predict/", QuitSmokingPredictAPIView.as_view(), name="api-quit-predict"),
    path("api/quit-smoking-history/", QuitSmokingHistoryAPIView.as_view(), name="api-quit-history"),

    # Login/Register API shortcut (opsiyonel)
    path("api/login/", login_page, name="api-login"),
    path("api/register/", register_page, name="api-register"),
]
