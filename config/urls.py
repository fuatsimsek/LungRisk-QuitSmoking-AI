from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # Tüm uygulama core.urls üzerinden yönetiliyor
    path("", include("core.urls")),
]
