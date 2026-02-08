# LungRisk & QuitSmoking AI

A dual-purpose web application that assesses **lung cancer risk** and **quit-smoking likelihood** using machine learning models. Users can sign up, submit health and lifestyle data, and receive instant risk analysis with personalized suggestions.

![Dashboard Result](screenshots/cancern3.jpg)

---

## 🎯 Project Purpose

This application serves two main predictive functions:

1.  **Lung Cancer Risk Assessment**: Predicts the risk level (**LOW / MEDIUM / HIGH**) based on factors like age, air pollution, fatigue, genetic risk, and alcohol use.
2.  **Quit-Smoking Likelihood**: Predicts the probability of a user successfully quitting smoking based on NHIS-style inputs (motivation, dependency, past attempts, etc.).

---

## 📸 Screenshots

### 1. Lung Cancer Risk Form
Users input their health data via interactive sliders to calculate cancer risk.
![Lung Cancer Risk Form](screenshots/cancern2.jpg)

### 2. Quit Smoking Likelihood Form
A specialized form predicting the probability of successfully quitting based on psychological and physical habits.
![Quit Smoking Form](screenshots/smoking2.png)

### 3. Results Dashboard
Instant prediction results with probability distribution and tailored advice.
![Results Dashboard](screenshots/cancern3.jpg)

### 4. Authentication
Secure Login and Registration pages.

| Login Screen | Register Screen |
| :---: | :---: |
| ![Login](screenshots/Login.png) | ![Register](screenshots/register.png) |

---

## 🧠 AI Models Used

### 1. Lung Cancer Risk
- **Model**: Multinomial **Logistic Regression** (scikit-learn).
- **Technique**: Class weighting used for balanced predictions.
- **Input**: Age (StandardScaled), lifestyle factors (MinMaxScaled).
- **Output**: Multi-class classification (Low, Medium, High).

### 2. Quit-Smoking Likelihood
- **Model**: **Random Forest Classifier** with probability calibration (`CalibratedClassifierCV`).
- **Technique**: **SMOTE** used to handle class imbalance; Mode/Median imputation for missing NHIS data.
- **Output**: Probability score (0-100%) and binary decision.

---

## 🛠 Tech Stack

- **Backend**: Django 5.x, Django REST Framework
- **Database**: PostgreSQL
- **ML Engine**: Scikit-learn, Pandas, Numpy, Joblib
- **Frontend**: Server-rendered HTML (Dark Theme), Vanilla JS
- **Authentication**: Session-based auth

---

## ⚙️ Installation and Run

### 1. Database Setup
Ensure PostgreSQL is running and create a database:
```sql
CREATE DATABASE lungrisk_db;
2. Configure Settings
Open config/settings.py and update the DATABASES section:

Python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'lungrisk_db',
        'USER': 'your_postgres_user',
        'PASSWORD': 'your_postgres_password',
        'HOST': '127.0.0.1',
        'PORT': '5432',
    }
}
3. Install & Start
Bash
# Create virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Apply migrations
python manage.py migrate

# Run server
python manage.py runserver
Visit http://127.0.0.1:8000/ to start using the app.