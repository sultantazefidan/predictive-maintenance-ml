import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.metrics import precision_recall_curve
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

# DATA
df = pd.read_csv(DATA_PATH)
print("\nDATASET INFO")

n_samples = df.shape[0]
n_features = df.shape[1] - 1  # target hariç
print("Toplam sütun sayısı (ham veri):", df.shape[1])

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

class_counts = df["Machine failure"].value_counts()
class_ratios = df["Machine failure"].value_counts(normalize=True) * 100

print(f"Total Samples        : {n_samples}")
print(f"Total Features       : {n_features}")
print(f"Dataset Size (MB)    : {memory_mb:.2f}")

print("\nClass Distribution:")
for cls in class_counts.index:
    print(
        f"  Class {cls} -> "
        f"{class_counts[cls]} samples "
        f"({class_ratios[cls]:.2f}%)"
    )
# Modele giren sütun isimleri (target hariç)
feature_columns = df.drop(columns=["Machine failure"]).columns

print("Modele giren sütunlar:")
for col in feature_columns:
    print(f" - {col}")


# özllik tbalosu
feature_df = df.drop(columns=["Machine failure"])
feature_table = []

for col in feature_df.columns:
    dtype = feature_df[col].dtype

    if dtype == "bool" or feature_df[col].nunique() == 2:
        feature_type = "Binary"
    elif dtype in ["int64", "float64"]:
        feature_type = "Numeric"
    else:
        feature_type = "Categorical"

    feature_table.append({
        "Feature Name": col,
        "Data Type": feature_type,
        "Original dtype": str(dtype)
    })

feature_table_df = pd.DataFrame(feature_table)

print("\nMODELE GİREN FEATURE TABLOSU:")
print(feature_table_df)
feature_table_df.to_markdown(
    os.path.join(OUTPUT_DIR, "feature_table.md"),
    index=False
)



print("------------------\n")


X = df.drop(columns=["Machine failure"]).values
y = df["Machine failure"].values


# Literatürde önerilen ve denenen nihai  model hiperparametrelerinin tanımlanması

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


# K-FOLD- tekrarlanabılırlık açısından r_state 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = defaultdict(list)


# TRAINING
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"--- Fold {fold}/5 ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for name, model in models.items():

        # TRAINING TIME - sadece trainde fit
        start_time = time.perf_counter()
        model.fit(X_train_s, y_train)
        train_time = time.perf_counter() - start_time

        # TRAIN
        y_train_pred = model.predict(X_train_s)
        y_train_prob = model.predict_proba(X_train_s)[:, 1]

        # TEST
        y_test_pred = model.predict(X_test_s)
        y_test_prob = model.predict_proba(X_test_s)[:, 1]

        # Specificity
        tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred).ravel()
        tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred).ravel()

        results[name].append({
            #TEST
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_specificity": tn_te / (tn_te + fp_te),
            "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            "test_auc": roc_auc_score(y_test, y_test_prob),

            #TRAIN
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_specificity": tn_tr / (tn_tr + fp_tr),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "train_auc": roc_auc_score(y_train, y_train_prob),

            #  TIME
            "train_time_sec": train_time,

            # RAW
            "y_test": y_test,
            "y_pred": y_test_pred,
            "y_prob": y_test_prob
        })


# nihai sonuçları tablolaştır
final_rows = []

for model_name, folds in results.items():
    df_folds = pd.DataFrame(folds)
    n_folds = len(df_folds)

    # TEST ROW
    final_rows.append({
        "Model": model_name,
        "Set": "Test",

        "Accuracy_mean": df_folds["test_accuracy"].mean(),
        "Accuracy_std": df_folds["test_accuracy"].std(),

        "Precision_mean": df_folds["test_precision"].mean(),
        "Precision_std": df_folds["test_precision"].std(),

        "Recall_mean": df_folds["test_recall"].mean(),
        "Recall_std": df_folds["test_recall"].std(),

        "Specificity_mean": df_folds["test_specificity"].mean(),
        "Specificity_std": df_folds["test_specificity"].std(),

        "F1_mean": df_folds["test_f1"].mean(),
        "F1_std": df_folds["test_f1"].std(),

        "AUC_mean": df_folds["test_auc"].mean(),
        "AUC_std": df_folds["test_auc"].std(),

        "TrainTime_mean_sec": np.nan,
        "TrainTime_std_sec": np.nan,

        "n_folds": n_folds
    })

    # TRAIN ROW
    final_rows.append({
        "Model": model_name,
        "Set": "Train",

        "Accuracy_mean": df_folds["train_accuracy"].mean(),
        "Accuracy_std": df_folds["train_accuracy"].std(),

        "Precision_mean": df_folds["train_precision"].mean(),
        "Precision_std": df_folds["train_precision"].std(),

        "Recall_mean": df_folds["train_recall"].mean(),
        "Recall_std": df_folds["train_recall"].std(),

        "Specificity_mean": df_folds["train_specificity"].mean(),
        "Specificity_std": df_folds["train_specificity"].std(),

        "F1_mean": df_folds["train_f1"].mean(),
        "F1_std": df_folds["train_f1"].std(),

        "AUC_mean": df_folds["train_auc"].mean(),
        "AUC_std": df_folds["train_auc"].std(),

        "TrainTime_mean_sec": df_folds["train_time_sec"].mean(),
        "TrainTime_std_sec": df_folds["train_time_sec"].std(),

        "n_folds": n_folds
    })

    # ROC & CM (TEST)
    y_true = np.concatenate(df_folds["y_test"].values)
    y_pred = np.concatenate(df_folds["y_pred"].values)
    y_prob = np.concatenate(df_folds["y_prob"].values)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.imshow(
        cm,
        cmap="Blues",
        interpolation="nearest",
        vmin=0,
        vmax=cm.max()
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]

            # Koyu hücre → açık yazı, açık hücre → koyu yazı
            text_color = "white" if value > max_val * 0.5 else "#08306b"

            plt.text(
                j, i, value,
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                #fontweight="bold"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.savefig(os.path.join(OUTPUT_DIR, f"ConfusionMatrix_{model_name}.png"), dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC - {model_name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"ROC_{model_name}.png"), dpi=300)
    plt.close()

    # precison & recall eğriler
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Precision Curve
    plt.figure()
    plt.plot(thresholds, precision[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title(f"Precision Curve - {model_name}")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"PrecisionCurve_{model_name}.png"), dpi=300)
    plt.close()

    # Recall Curve
    plt.figure()
    plt.plot(thresholds, recall[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title(f"Recall Curve - {model_name}")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"RecallCurve_{model_name}.png"), dpi=300)
    plt.close()


# Excele kaydet
final_df = pd.DataFrame(final_rows).round(4)

xlsx_path = os.path.join(
    OUTPUT_DIR,
    "ai4i+2020+predictive+maintenance_results_train_test_mean_std_WITH_TIME.xlsx"
)

final_df.to_excel(xlsx_path, index=False)

print("\nTÜM MODELLER BAŞARIYLA ÇALIŞTI")
print("Excel çıktı:", xlsx_path)
print("Grafikler:", OUTPUT_DIR)
