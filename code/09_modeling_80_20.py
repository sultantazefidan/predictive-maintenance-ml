import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Dosya yolları
BASE_DIR = r"C:\Users\Gaming\Desktop\ai4i+2020+predictive+maintenance+dataset"
DATA_PATH = os.path.join(BASE_DIR, "ai4i2020_clean_step2.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Machine failure"]).values
y = df["Machine failure"].values


# Train-Test ayırması
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#Standardscaler ile normalize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


#modellerin literatüre uygun hiperparametreleirin ayrlanması
models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

results = []


# eğitim ve değerlendirme süreci
for name, model in models.items():

    print(f"Training {name}...")

    start = time.perf_counter()
    model.fit(X_train_s, y_train)
    train_time = time.perf_counter() - start

    # TRAIN
    y_train_pred = model.predict(X_train_s)
    y_train_prob = model.predict_proba(X_train_s)[:, 1]

    # TEST
    y_test_pred = model.predict(X_test_s)
    y_test_prob = model.predict_proba(X_test_s)[:, 1]

    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred).ravel()
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred).ravel()

    results.append({
        "Model": name,

        # test
        "Test_Accuracy": accuracy_score(y_test, y_test_pred),
        "Test_Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Test_Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "Test_Specificity": tn_te / (tn_te + fp_te),
        "Test_F1": f1_score(y_test, y_test_pred, zero_division=0),
        "Test_AUC": roc_auc_score(y_test, y_test_prob),

        # train
        "Train_Accuracy": accuracy_score(y_train, y_train_pred),
        "Train_Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Train_Recall": recall_score(y_train, y_train_pred, zero_division=0),
        "Train_Specificity": tn_tr / (tn_tr + fp_tr),
        "Train_F1": f1_score(y_train, y_train_pred, zero_division=0),
        "Train_AUC": roc_auc_score(y_train, y_train_prob),

        "Train_Time_sec": train_time
    })

    #ROC
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC - {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"ROC_{name}.png"), dpi=300)
    plt.close()

    #CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(OUTPUT_DIR, f"ConfusionMatrix_{name}.png"), dpi=300)
    plt.close()

    # PRECISION–RECALL
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)

    plt.figure()
    plt.plot(thresholds, precision[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title(f"Precision Curve - {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"PrecisionCurve_{name}.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(thresholds, recall[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title(f"Recall Curve - {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"RecallCurve_{name}.png"), dpi=300)
    plt.close()


# excele kaydet
final_df = pd.DataFrame(results).round(4)

xlsx_path = os.path.join(
    OUTPUT_DIR,
    "AI4I_80_20_results.xlsx"
)

final_df.to_excel(xlsx_path, index=False)

print("\n TÜM MODELLER TAMAMLANDI")
print("Excel:", xlsx_path)
print("Grafikler:", OUTPUT_DIR)
