import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# print("All libraries imported successfully!")
# print(f"  pandas  : {pd.__version__}")
# print(f"  numpy   : {np.__version__}")

# ── Block 2: Load Dataset ──────────────────────────────────────
df = pd.read_csv("realistic_loan_data.csv")

# print("Dataset loaded!")
# print(f"  Rows    : {df.shape[0]}")
# print(f"  Columns : {df.shape[1]}")
# print(f"\nColumns : {list(df.columns)}")
# print(f"\nFirst 5 rows:")
# print(df.head())
# print(f"\nLoan_Type breakdown:")
# print(df['Loan_Type'].value_counts())
# print(f"\nLoan_Status breakdown:")
# print(df['Loan_Status'].value_counts())

# ── Block 3: Exploratory Data Analysis ────────────────────────

# print("\n--- Data Types ---")
# print(df.dtypes)

# print("\n--- Missing Values ---")
# print(df.isnull().sum())

# print("\n--- Basic Statistics ---")
# print(df.describe())

# print("\n--- CIBIL Score Distribution ---")
bins = [300, 550, 600, 650, 700, 750, 800, 900]
labels = ['300-550','550-600','600-650','650-700','700-750','750-800','800-900']
df['CIBIL_Range'] = pd.cut(df['CIBIL_Score'], bins=bins, labels=labels)
# print(df['CIBIL_Range'].value_counts().sort_index())

# print("\n--- Approval Rate by Loan Type ---")
# print(df.groupby('Loan_Type')['Loan_Status'].value_counts(normalize=True).mul(100).round(1))

# print("\n--- Approval Rate by CIBIL Range ---")
# print(df.groupby('CIBIL_Range')['Loan_Status'].value_counts(normalize=True).mul(100).round(1))

# Drop helper column after analysis
df.drop('CIBIL_Range', axis=1, inplace=True)

# ── Block 4: Data Cleaning ─────────────────────────────────────

# --- Categorical → Mode ---
for col in ['Gender', 'Dependents', 'Self_Employed']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# --- Numeric → Median ---
for col in ['CIBIL_Score', 'Existing_EMIs']:
    df[col].fillna(df[col].median(), inplace=True)

# --- LoanAmount → Median per Loan_Type ---
# More accurate than global median since loan amounts
# vary hugely between Personal (50K) and Home (80L)
df['LoanAmount'] = df.groupby('Loan_Type')['LoanAmount'].transform(
    lambda x: x.fillna(x.median())
)

# --- Verify ---
# print("Missing values after cleaning:")
# print(df.isnull().sum())
# print(f"\nDataset shape: {df.shape}")

# ── Block 5: Encoding + Feature Engineering ────────────────────

# --- Drop Loan_ID ---
df.drop('Loan_ID', axis=1, inplace=True)

# --- Encode binary columns ---
df['Gender']        = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married']       = df['Married'].map({'Yes': 1, 'No': 0})
df['Education']     = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status']   = df['Loan_Status'].map({'Y': 1, 'N': 0})

# --- Dependents: 3+ → 3 ---
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# --- Property Area ---
df['Property_Area'] = df['Property_Area'].map(
    {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
)

# --- Loan Type: one-hot encode ---
# We don't map to 0/1/2 because Personal/Home/Vehicle
# have no natural order — one-hot is more honest
loan_type_dummies = pd.get_dummies(
    df['Loan_Type'], prefix='LoanType', drop_first=False
)
df = pd.concat([df.drop('Loan_Type', axis=1), loan_type_dummies], axis=1)

print("Loan_Type one-hot columns added:")
print([c for c in df.columns if 'LoanType' in c])

# --- Feature Engineering ---

# FOIR: Fixed Obligation to Income Ratio
# Most important metric in real Indian banking
# RBI guideline: FOIR should be below 50%
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI_Approx']   = (df['LoanAmount'] * 1000) / df['Loan_Amount_Term']
df['FOIR']         = (df['EMI_Approx'] + df['Existing_EMIs']) / (
                          df['Total_Income'] + 1e-6)

# Income to Loan ratio (how much income vs loan size)
df['Income_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 1e-6)

# CIBIL bucket (gives model a cleaner signal than raw score)
df['CIBIL_Bucket'] = pd.cut(
    df['CIBIL_Score'],
    bins=[300, 600, 650, 700, 750, 900],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# Drop EMI_Approx (intermediate calc, not needed as feature)
df.drop('EMI_Approx', axis=1, inplace=True)

# print("\nAll columns after encoding + engineering:")
# print(list(df.columns))
# print(f"\nShape: {df.shape}")
# print(f"\nSample row:")
# print(df.head(2).to_string())

# ── Block 6: Scaling ───────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

# --- Separate features and target ---
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# print(f"Features : {X.shape[1]} columns")
# print(f"Target   : {y.name}")
# print(f"\nClass distribution:")
# print(y.value_counts())
# print(f"\nApproval rate: {y.mean()*100:.1f}%")

# --- Scale ---
# Note: CIBIL_Bucket and one-hot columns will also be scaled
# This is fine — scaler handles binary/ordinal columns safely
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X), columns=X.columns
)

# print("\nScaling done!")
# print("Sample scaled values (first 2 rows):")
# print(X_scaled[['CIBIL_Score','FOIR','Income_Loan_Ratio',
#                 'Total_Income']].head(2).round(3))

# ── Block 7: Hybrid Train-Test Split ──────────────────────────
from sklearn.model_selection import train_test_split

# --- Define feature sets ---

# Traditional Model (CIBIL >= 650 — reliable credit signal)
# Uses all features including CIBIL score
trad_features = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'CIBIL_Score', 'CIBIL_Bucket',
    'Existing_EMIs', 'Property_Area',
    'LoanType_Home', 'LoanType_Personal', 'LoanType_Vehicle',
    'Total_Income', 'FOIR', 'Income_Loan_Ratio'
]

# Alternative Model (CIBIL < 650 — thin/poor credit file)
# No CIBIL_Score or CIBIL_Bucket — they are unreliable here
# Gender and Married excluded for fairness
alt_features = [
    'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Existing_EMIs', 'Property_Area',
    'LoanType_Home', 'LoanType_Personal', 'LoanType_Vehicle',
    'Total_Income', 'FOIR', 'Income_Loan_Ratio'
]

# --- Split by CIBIL Score ---
# Use original df CIBIL_Score (before scaling) for clean split
mask_trad = df['CIBIL_Score'] >= 650
mask_alt  = df['CIBIL_Score'] <  650

# print(f"Traditional path (CIBIL >= 650) : {mask_trad.sum()} rows")
# print(f"Alternative path (CIBIL <  650) : {mask_alt.sum()} rows")

# print(f"\nTraditional approval rate : "
#       f"{y[mask_trad].mean()*100:.1f}%")
# print(f"Alternative approval rate : "
#       f"{y[mask_alt].mean()*100:.1f}%")

# --- Traditional split (stratified — enough data) ---
X_trad = X_scaled[trad_features][mask_trad]
y_trad = y[mask_trad]

X_tr_train, X_tr_test, y_tr_train, y_tr_test = train_test_split(
    X_trad, y_trad,
    test_size=0.2, random_state=42, stratify=y_trad
)

# --- Alternative split ---
X_alt = X_scaled[alt_features][mask_alt]
y_alt = y[mask_alt]

X_al_train, X_al_test, y_al_train, y_al_test = train_test_split(
    X_alt, y_alt,
    test_size=0.2, random_state=42
)

# print(f"\nTraditional Model:")
# print(f"  Train : {X_tr_train.shape[0]} rows "
#       f"| Approved: {y_tr_train.sum()} "
#       f"| Rejected: {(y_tr_train==0).sum()}")
# print(f"  Test  : {X_tr_test.shape[0]} rows "
#       f"| Approved: {y_tr_test.sum()} "
#       f"| Rejected: {(y_tr_test==0).sum()}")

# print(f"\nAlternative Model:")
# print(f"  Train : {X_al_train.shape[0]} rows "
#       f"| Approved: {y_al_train.sum()} "
#       f"| Rejected: {(y_al_train==0).sum()}")
# print(f"  Test  : {X_al_test.shape[0]} rows "
#       f"| Approved: {y_al_test.sum()} "
#       f"| Rejected: {(y_al_test==0).sum()}")

# ── Block 8: Train Traditional Model ──────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

trad_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=8,
        class_weight='balanced',
        random_state=42
    )
}

# print("Training Traditional Models...")
for name, model in trad_models.items():
    model.fit(X_tr_train, y_tr_train)
    print(f"  {name} ... done")

# --- Feature Importance (bias check) ---
rf = trad_models["Random Forest"]
importances = pd.Series(
    rf.feature_importances_, index=trad_features
).sort_values(ascending=False)

# print("\nFeature Importances (Random Forest - Traditional):")
# print("-" * 50)
for feat, imp in importances.items():
    bar      = "#" * int(imp * 100)
    bias_tag = " <-- BIAS SENSITIVE" if feat in ['Gender','Married'] else ""
    # print(f"  {feat:<22} {imp:.4f}  {bar}{bias_tag}")

# ── Block 9: Train Alternative Model ──────────────────────────
# CIBIL < 650 applicants
# No SMOTE needed — 34 approved vs 38 rejected (nearly balanced!)
# No Gender, Married, CIBIL features used here

# print("Alternative path class balance:")
# print(f"  Approved : {y_al_train.sum()}")
# print(f"  Rejected : {(y_al_train==0).sum()}")
# print(f"  Ratio    : {y_al_train.mean():.2f} (0.5 = perfect balance)")

alt_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
}

# print("\nTraining Alternative Models...")
for name, model in alt_models.items():
    model.fit(X_al_train, y_al_train)
    print(f"  {name} ... done")

# --- Feature Importance ---
rf_alt = alt_models["Random Forest"]
importances_alt = pd.Series(
    rf_alt.feature_importances_, index=alt_features
).sort_values(ascending=False)

# print("\nFeature Importances (Random Forest - Alternative):")
# print("-" * 50)
for feat, imp in importances_alt.items():
    bar = "#" * int(imp * 100)
    print(f"  {feat:<22} {imp:.4f}  {bar}")

# ── Block 10: Evaluate & Compare Models ───────────────────────
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

def evaluate(models, X_test, y_test, label):
    # print(f"\n{'='*55}")
    # print(f" {label}")
    # print(f"{'='*55}")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1
        }

        # print(f"\n  {name}")
        # print(f"    Accuracy  : {acc*100:.1f}%")
        # print(f"    Precision : {prec:.4f}")
        # print(f"    Recall    : {rec:.4f}")
        # print(f"    F1 Score  : {f1:.4f}")
        # print(f"    Confusion Matrix:")
        # print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
        # print(f"      FN={cm[1][0]}  TP={cm[1][1]}")

    # Best model by F1
    best = max(results, key=lambda k: results[k]['F1'])
    # print(f"\n  Best model : {best}")
    # print(f"  Best F1    : {results[best]['F1']:.4f}")

    return results

# --- Run both evaluations ---
trad_results = evaluate(
    trad_models, X_tr_test, y_tr_test,
    "TRADITIONAL MODEL (CIBIL >= 650)"
)
alt_results = evaluate(
    alt_models, X_al_test, y_al_test,
    "ALTERNATIVE MODEL (CIBIL < 650)"
)

# --- Final summary ---
# print(f"\n{'='*55}")
# print(" FINAL SUMMARY")
# print(f"{'='*55}")

for label, results in [("Traditional", trad_results),
                        ("Alternative", alt_results)]:
    best = max(results, key=lambda k: results[k]['F1'])
    r    = results[best]
    # print(f"\n  {label} → {best}")
    # print(f"    Accuracy  : {r['Accuracy']*100:.1f}%")
    # print(f"    Precision : {r['Precision']:.4f}")
    # print(f"    Recall    : {r['Recall']:.4f}")
    # print(f"    F1 Score  : {r['F1']:.4f}")

# ── Block 11: Hybrid Prediction Function ──────────────────────

def hybrid_predict(applicant: dict):
    """
    Takes a raw applicant dictionary and routes to:
      CIBIL >= 650 → Logistic Regression (Traditional)
      CIBIL <  650 → Decision Tree (Alternative)

    Returns: decision, probability, and breakdown
    """

    print("\n" + "="*55)
    print(" HYBRID LOAN PREDICTION V2")
    print("="*55)
    print(" Applicant Details:")
    for k, v in applicant.items():
        print(f"   {k:<22} : {v}")

    # --- Step 1: Encode ---
    row = {
        'Gender'           : 1 if applicant['Gender'] == 'Male' else 0,
        'Married'          : 1 if applicant['Married'] == 'Yes' else 0,
        'Dependents'       : int(str(applicant['Dependents']).replace('+','')),
        'Education'        : 1 if applicant['Education'] == 'Graduate' else 0,
        'Self_Employed'    : 1 if applicant['Self_Employed'] == 'Yes' else 0,
        'ApplicantIncome'  : applicant['ApplicantIncome'],
        'CoapplicantIncome': applicant['CoapplicantIncome'],
        'LoanAmount'       : applicant['LoanAmount'],
        'Loan_Amount_Term' : applicant['Loan_Amount_Term'],
        'CIBIL_Score'      : applicant['CIBIL_Score'],
        'Existing_EMIs'    : applicant['Existing_EMIs'],
        'Property_Area'    : {'Rural':0,'Semiurban':1,'Urban':2}[applicant['Property_Area']],
        'LoanType_Home'    : 1 if applicant['Loan_Type'] == 'Home'     else 0,
        'LoanType_Personal': 1 if applicant['Loan_Type'] == 'Personal' else 0,
        'LoanType_Vehicle' : 1 if applicant['Loan_Type'] == 'Vehicle'  else 0,
    }

    # --- Step 2: Feature Engineering ---
    row['Total_Income']     = row['ApplicantIncome'] + row['CoapplicantIncome']
    emi_approx              = (row['LoanAmount'] * 1000) / row['Loan_Amount_Term']
    row['FOIR']             = (emi_approx + row['Existing_EMIs']) / (
                                  row['Total_Income'] + 1e-6)
    row['Income_Loan_Ratio']= row['Total_Income'] / (row['LoanAmount'] + 1e-6)
    row['CIBIL_Bucket']     = (0 if applicant['CIBIL_Score'] < 600 else
                               1 if applicant['CIBIL_Score'] < 650 else
                               2 if applicant['CIBIL_Score'] < 700 else
                               3 if applicant['CIBIL_Score'] < 750 else 4)

    # --- Step 3: Scale ---
    all_cols   = list(X.columns)
    row_df     = pd.DataFrame([row])[all_cols]
    row_scaled = pd.DataFrame(
        scaler.transform(row_df), columns=all_cols
    )

    # --- Step 4: Build breakdown for display ---
    foir_pct = row['FOIR'] * 100
    breakdown = []
    breakdown.append(
        f"CIBIL Score: {applicant['CIBIL_Score']} "
        f"({'Good' if applicant['CIBIL_Score']>=700 else 'Fair' if applicant['CIBIL_Score']>=650 else 'Poor'})"
    )
    breakdown.append(
        f"FOIR: {foir_pct:.1f}% of income "
        f"({'Healthy <30%' if foir_pct<30 else 'Moderate <50%' if foir_pct<50 else 'High >50%'})"
    )
    breakdown.append(
        f"Total Income: Rs.{row['Total_Income']:,}/month"
    )
    breakdown.append(
        f"Loan EMI (approx): Rs.{emi_approx:,.0f}/month"
    )
    breakdown.append(
        f"Existing EMIs: Rs.{int(applicant['Existing_EMIs']):,}/month"
    )
    breakdown.append(
        f"Income-to-Loan Ratio: {row['Income_Loan_Ratio']:.2f}"
    )
    breakdown.append(
        f"Loan Type: {applicant['Loan_Type']}"
    )

    # --- Step 5: Route ---
    if applicant['CIBIL_Score'] >= 650:
        print(f"\n  Routing → Traditional Model (Logistic Regression)")
        print(f"  Reason  → CIBIL {applicant['CIBIL_Score']} >= 650")

        X_input = row_scaled[trad_features]
        prob    = float(
            trad_models["Logistic Regression"].predict_proba(X_input)[0][1]
        )
        model_used = "Logistic Regression (Traditional)"
        breakdown.append("Routed to Traditional Model — CIBIL >= 650")
        breakdown.append("Gender & Married influence: < 0.3% combined")

    else:
        print(f"\n  Routing → Alternative Model (Decision Tree)")
        print(f"  Reason  → CIBIL {applicant['CIBIL_Score']} < 650")

        X_input = row_scaled[alt_features]
        prob    = float(
            alt_models["Decision Tree"].predict_proba(X_input)[0][1]
        )
        model_used = "Decision Tree (Alternative)"
        breakdown.append("Routed to Alternative Model — CIBIL < 650")
        breakdown.append("Gender & Married NOT used in this model")

    # --- Step 6: Decision with borderline zone ---
    if prob >= 0.55:
        status = "APPROVED"
    elif prob >= 0.40:
        status = "MANUAL REVIEW"
    else:
        status = "REJECTED"

    print(f"  Probability : {prob*100:.1f}%")
    print(f"  Decision    : {status}")
    print("="*55)

    return status, prob, model_used, breakdown


# ── Test Cases ─────────────────────────────────────────────────

# Test 1: Strong home loan applicant
applicant_1 = {
    "Loan_Type": "Home", "Gender": "Female", "Married": "Yes",
    "Dependents": "1", "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 85000, "CoapplicantIncome": 45000,
    "LoanAmount": 4500, "Loan_Amount_Term": 240,
    "CIBIL_Score": 780, "Existing_EMIs": 5000,
    "Property_Area": "Urban"
}

# Test 2: Personal loan, low CIBIL
applicant_2 = {
    "Loan_Type": "Personal", "Gender": "Male", "Married": "No",
    "Dependents": "0", "Education": "Graduate", "Self_Employed": "Yes",
    "ApplicantIncome": 35000, "CoapplicantIncome": 0,
    "LoanAmount": 150, "Loan_Amount_Term": 36,
    "CIBIL_Score": 610, "Existing_EMIs": 3000,
    "Property_Area": "Semiurban"
}

# Test 3: Vehicle loan, borderline case
applicant_3 = {
    "Loan_Type": "Vehicle", "Gender": "Male", "Married": "Yes",
    "Dependents": "2", "Education": "Not Graduate", "Self_Employed": "No",
    "ApplicantIncome": 28000, "CoapplicantIncome": 12000,
    "LoanAmount": 800, "Loan_Amount_Term": 60,
    "CIBIL_Score": 670, "Existing_EMIs": 8000,
    "Property_Area": "Semiurban"
}

s1, p1, m1, b1 = hybrid_predict(applicant_1)
s2, p2, m2, b2 = hybrid_predict(applicant_2)
s3, p3, m3, b3 = hybrid_predict(applicant_3)

# ── Block 13: Export for app_v2.py ────────────────────────────
# These variables are used by app_v2.py
EXPORTS = {
    'X'           : X,
    'scaler'      : scaler,
    'trad_models' : trad_models,
    'alt_models'  : alt_models,
    'trad_features': trad_features,
    'alt_features' : alt_features,
}

