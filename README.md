# Tourism Accommodation Grading ML Project 🏨

## Overview
Tourism is a major economic sector in Sri Lanka. Accommodation grading plays an important role in helping tourism authorities, businesses, and investors make data-driven decisions. This project develops a **Machine Learning model** to predict the **grade level** of tourism accommodations as **High, Medium, or Low** based on features like:

- Number of rooms (`Rooms`)
- Geographic location (`Latitude`, `Longitude`)
- District (`District`)
- Property type (`Type`) – e.g., Boutique Hotels, Villas, Bungalows, Guest Houses

The model helps business owners, investors, and authorities understand market positioning, pricing strategies, and expected standards for accommodations.

---

## Problem Statement
- Manual grading is **time-consuming, inconsistent, and subjective**.  
- Businesses face uncertainty in pricing and branding decisions.  
- Investors lack predictive tools to estimate the grade level before development.  
- Tourism authorities need to **allocate inspections efficiently** while maintaining standards.  

This project solves these problems by **automatically predicting accommodation grades** and providing explainable insights for decision-making.

---

## Dataset
- **Source:** Collected local dataset of Sri Lankan accommodations (`Accommodation.csv`)  
- **Features Used:**  
  - `Rooms` – Number of rooms in the property  
  - `Latitude` & `Longitude` – Geographic coordinates  
  - `District` – Administrative district  
  - `Type` – Accommodation type  
- **Target Variable:** `Grade` mapped to 3 categories:
  - **High** – Deluxe, Superior, Five  
  - **Medium** – Four, Standard, Three  
  - **Low** – One, Two, Basic  

- **Preprocessing:**
  - Removed duplicates and missing values  
  - Encoded categorical variables (One-Hot Encoding)  
  - Scaled numerical features (StandardScaler)  
  - Encoded target variable (LabelEncoder)

---

## Machine Learning Model
- **Algorithm Used:** Extra Trees Classifier  
- **Why Extra Trees?**
  - Reduces variance with multiple random trees  
  - Handles **non-linear relationships**  
  - Works well with **mixed numeric and categorical data**  
  - Robust against **overfitting**  

- **Difference from Other Models:**
  - **Decision Trees:** Single tree, higher variance, less robust  
  - **Random Forest:** Extra Trees introduces more randomness for better generalization  
  - **Logistic Regression:** Linear model, cannot capture complex non-linear patterns  

- **Training & Evaluation:**
  - Train-test split: 70%-30%  
  - 5-fold cross-validation  
  - Metrics: Accuracy, Classification Report, Confusion Matrix  
  - SHAP analysis used for feature importance and explainability  

---

## Feature Importance
The model identified the most important features for predicting grade:

1. `Rooms` – Larger properties tend to be higher grade  
2. `District` – Urban districts like Colombo correlate with higher grades  
3. `Type` – Property type influences predicted grade  
4. `Latitude` & `Longitude` – Geographical location impacts grading  

---

## Critical Discussion
**Limitations**
- Dataset size may limit generalization across all Sri Lanka  
- Latitude and Longitude alone do not capture location quality (proximity to attractions, roads, safety)  
- Only five features used, service quality and amenities not included  

**Bias Risks**
- Certain districts dominate the dataset, causing geographic bias  
- Mapping detailed grades to 3 categories oversimplifies grading  

**Ethical Considerations**
- No personal data used, privacy risk is minimal  
- Predictions are **decision support only**, not a replacement for human inspections  

---

## Project Structure

tourism_ml_project/
│
├── data/
│ └── Accommodation.csv
│
├── models/
│ ├── model.pkl
│ └── target_encoder.pkl
│
├── train.py # Train ML model, SHAP analysis, save model
├── app.py # FastAPI backend (optional)
├── streamlit_app.py # Streamlit frontend
├── README.md
└── requirements.txt


## How to Use
### Clone the repo
```bash
git clone https://github.com/GHBhaumika/tourism-accommodation-grading-ml.git
cd tourism-accommodation-grading-ml

0️⃣ Setup Virtual Environment 
# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux / Mac)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

1️⃣ Train the Model
python train.py
This will preprocess the dataset, train the Extra Trees classifier, generate SHAP plots, and save the model and encoder in models/.

2️⃣ Run Streamlit App (Frontend)
streamlit run streamlit_app.py

3️⃣ Run FastAPI (Backend API)
uvicorn app:app --reload
Access API at http://127.0.0.1:8000
POST /predict endpoint to get grade prediction

Acknowledgements
Dataset collected locally from Sri Lankan tourism accommodations
Python libraries used: pandas, scikit-learn, shap, streamlit, fastapi, matplotlib, seaborn
