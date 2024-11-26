from flask import Flask, render_template, request
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

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def hello_world():
    title = "Cancer Detection"
    
    

    new_patient_data = []

    prediction = None
    prediction_proba = None
    benign_metrics = None
    malignant_metrics = None
    accuracy = None
    macro_avg = None
    weighted_avg = None

    if request.method == "POST":

        mean_radius =  float(request.form["mean_radius"])
        mean_texture = float(request.form["mean_texture"])
        mean_perimeter = float(request.form["mean_perimeter"])
        mean_area = float(request.form["mean_area"])
        mean_smoothness = float(request.form["mean_smoothness"])
        mean_compactness = float(request.form["mean_compactness"])
        mean_concavity = float(request.form["mean_concavity"])
        mean_concave_points = float(request.form["mean_concave_points"])
        mean_symmetry = float(request.form["mean_symmetry"])
        mean_fractal_dimension = float(request.form["mean_fractal_dimension"])
        radius_error = float(request.form["radius_error"])
        texture_error = float(request.form["texture_error"])
        perimeter_error = float(request.form["perimeter_error"])
        area_error = float(request.form["area_error"])
        smoothness_error = float(request.form["smoothness_error"])
        compactness_error = float(request.form["compactness_error"])
        concavity_error = float(request.form["concavity_error"])
        concave_points_error = float(request.form["concave_points_error"])
        symmetry_error = float(request.form["symmetry_error"])
        fractal_dimension_error = float(request.form["fractal_dimension_error"])
        worst_radius = float(request.form["worst_radius"])
        worst_texture = float(request.form["worst_texture"])
        worst_perimeter = float(request.form["worst_perimeter"])
        worst_area = float(request.form["worst_area"])
        worst_smoothness = float(request.form["worst_smoothness"])
        worst_compactness = float(request.form["worst_compactness"])
        worst_concavity = float(request.form["worst_concavity"])
        worst_concave_points = float(request.form["worst_concave_points"])
        worst_symmetry = float(request.form["worst_symmetry"])
        worst_fractal_dimension = float(request.form["worst_fractal_dimension"])


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

        # Mendapatkan classification report dalam bentuk dictionary
        classification_report_dict = classification_report(
            y_test, y_pred, target_names=['Benign', 'Malignant'], output_dict=True
        )

        # Contoh akses data dari classification_report_dict
        benign_metrics = classification_report_dict['Benign']
        malignant_metrics = classification_report_dict['Malignant']
        accuracy = classification_report_dict['accuracy']
        macro_avg = classification_report_dict['macro avg']
        weighted_avg = classification_report_dict['weighted avg']

        print("\nClassification Report dalam Format Dictionary:")
        print(classification_report_dict)

        new_patient_data = [[
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
            radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
        ]]

        prediction = rf_model.predict(new_patient_data)
        prediction_proba = rf_model.predict_proba(new_patient_data)

        prob_benign = prediction_proba[0][0] * 100
        prob_malignant = prediction_proba[0][1] * 100

        if prediction == 0:
            prediction = "Benign"
            prediction_proba = prob_benign
        else:
            prediction = "Malignant"
            prediction_proba = prob_malignant

    return render_template("home.html", title=title, benign_metrics=benign_metrics, malignant_metrics=malignant_metrics, accuracy=accuracy, macro_avg=macro_avg, weighted_avg=weighted_avg, prediction=prediction, prediction_proba=prediction_proba)