import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
file_path = 'cancer_classification.csv'
data = pd.read_csv(file_path)

# 2. Data Preprocessing
X = data.drop(columns=['benign_0__mal_1'])
y = data['benign_0__mal_1']

# 3. Pembagian Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Klasifikasi - Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 5. Prediksi dan Evaluasi
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# 6. Metrik Evaluasi Mendetail
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary'
)
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# 7. Visualisasi Metrik Performansi
plt.figure(figsize=(15, 5))

# Subplot 1: Bar Chart Metrik
plt.subplot(131)
metrics = ['Precision', 'Recall', 'F1-Score', 'Akurasi', 'AUC-ROC']
values = [precision, recall, f1_score, accuracy, auc_roc]
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.title('Performa Model Klasifikasi')
plt.ylabel('Skor')
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

# Subplot 2: Confusion Matrix
plt.subplot(132)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')

# Subplot 3: ROC Curve
plt.subplot(133)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title(f'ROC Curve (AUC = {auc_roc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.tight_layout()
plt.show()

# # 8. Cetak Laporan Klasifikasi
# print("\nClassification Report Lengkap:")
# print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# 9. Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 10. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Fitur Terpenting')
plt.show()

# 11. Prediksi Pasien Baru
new_patient_data = [[30, 0, 10.0, 80, 0.02, 20, 0.01, 0.02, 0.1,
                     0.05, 0.02, 0.05, 0.01, 0.1, 0.05, 0.5, 0.3, 10.0,
                     80, 0, 0.1, 0.05, 0.02, 0.05, 0.01, 0.1, 0.05,
                     0.02, 0.05, 0.01]]

prediction = rf_model.predict(new_patient_data)
prediction_proba = rf_model.predict_proba(new_patient_data)

prob_benign = prediction_proba[0][0] * 100
prob_malignant = prediction_proba[0][1] * 100

print("\nHasil Prediksi untuk Pasien Baru:")
if prob_benign > prob_malignant:
    print(f"Prediksi: Tumor Jinak (Benign) dengan probabilitas {prob_benign:.2f}%")
    print(f"Probabilitas Tumor Ganas (Malignant): {prob_malignant:.2f}%")
else:
    print(f"Prediksi: Tumor Ganas (Malignant) dengan probabilitas {prob_malignant:.2f}%")
    print(f"Probabilitas Tumor Jinak (Benign): {prob_benign:.2f}%")