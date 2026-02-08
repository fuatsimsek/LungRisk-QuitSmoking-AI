from rest_framework import serializers

from django.contrib.auth.models import User
from .models import CancerRiskRecord


class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["email", "password"]
        extra_kwargs = {
            "password": {"write_only": True},
        }

    def create(self, validated_data):
        raw_password = validated_data.pop("password")
        email = validated_data.pop("email")
        # Django User requires username; we use email as username.
        user = User.objects.create_user(
            username=email,
            email=email,
            password=raw_password,
            **validated_data,
        )
        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        email = data.get("email")
        password = data.get("password")

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise serializers.ValidationError("User not found")

        if not user.check_password(password):
            raise serializers.ValidationError("Wrong password")

        data["user"] = user
        return data


class CancerRiskRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = CancerRiskRecord
        fields = [
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
            "created_at",
        ]