# LungRisk & QuitSmoking AI

A dual-purpose web application that assesses **lung cancer risk** and **quit-smoking likelihood** using machine learning models. Users can sign up, submit health and lifestyle data, and receive instant risk analysis with personalized suggestions.

---

## 🎯 Project Purpose

This application serves two main predictive functions:

1.  **🫁 Lung Cancer Risk Assessment**: Predicts the risk level (**LOW / MEDIUM / HIGH**) based on factors like age, air pollution, fatigue, genetic risk, and alcohol use.
2.  **🚬 Quit-Smoking Likelihood**: Predicts the probability of a user successfully quitting smoking based on NHIS-style inputs (motivation, dependency, past attempts, etc.).

---

## 📸 Screenshots

### 1. Lung Cancer Analysis Module
Users input their health data via interactive sliders to calculate cancer risk.

| **Input Form** | **Result Dashboard** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/faa02deb-92cc-4c66-9ee4-c4429e69b8a4" width="450" alt="Lung Risk Form"> | <img src="https://github.com/user-attachments/assets/43ce8b48-2f70-4763-b1d4-26fc4faaadba" width="450" alt="Lung Risk Result"> |

### 2. Quit Smoking Analysis Module
A specialized form predicting the probability of successfully quitting based on psychological and physical habits.

| **Input Form** | **Result Dashboard** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/f084ad51-3cae-4dc3-99cd-29c63cdf0884" width="450" alt="Quit Form"> | <img src="https://github.com/user-attachments/assets/fe0d9c00-ddcd-4045-ac65-e754698918a2" width="450" alt="Quit Result"> |

### 3. Authentication
Secure Login and Registration pages with modern UI.

| **Login Screen** | **Register Screen** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/33ef5c67-c38f-4006-9cd5-e0988e9bce40" width="300" alt="Login"> | <img src="https://github.com/user-attachments/assets/dbb2b1be-f832-4803-acdc-af9298a7f13b" width="350" alt="Register"> |

---

## ⚠️ Important: Installation Note (Model Extraction)

**CRITICAL STEP:** Because the Random Forest model file is large (>100MB), it is compressed in the repository to avoid GitHub limits. **You must extract it before running the app.**

1.  Navigate to the `ml2/` folder in your project directory.
2.  Find the file named **`ml2.rar`** (or `.zip`).
3.  Right-click and select **"Extract Here"**.
4.  Ensure the file `model_quityrs_rf_calibrated.pkl` appears in the `ml2/` folder.

---

## 🧠 AI Models Used

### 1. Lung Cancer Risk
- **Model**: Multinomial **Logistic Regression** (scikit-learn).
- **Technique**: Class weighting used for balanced predictions across risk levels.
- **Input**: Age (StandardScaled), lifestyle factors like smoking and pollution (MinMaxScaled).
- **Output**: Multi-class classification (Low, Medium, High).

### 2. Quit-Smoking Likelihood
- **Model**: **Random Forest Classifier** with probability calibration (`CalibratedClassifierCV`).
- **Technique**: **SMOTE** used to handle class imbalance; Mode/Median imputation for missing NHIS data.
- **Output**: Probability score (0-100%) and binary decision (Likely/Unlikely).

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
