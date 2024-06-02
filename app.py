from flask import Flask, request, render_template, jsonify
from flask_wtf import FlaskForm
from wtforms import SubmitField, FloatField
from wtforms.validators import DataRequired
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import pandas as pd
import os
import joblib

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Clé secrète pour les formulaires WTForms

# Définir le chemin du projet et initialiser Kedro
project_path = Path(__file__).resolve().parent
metadata = bootstrap_project(project_path)

# Chemin vers le modèle final
model_path = project_path / 'data/06_models/model_final.pkl'

class PredictionForm(FlaskForm):
    """
    Formulaire Flask-WTF pour les prédictions audiométriques.
    """
    before_exam_125_Hz = FloatField('125 Hz', validators=[DataRequired()])
    before_exam_250_Hz = FloatField('250 Hz', validators=[DataRequired()])
    before_exam_500_Hz = FloatField('500 Hz', validators=[DataRequired()])
    before_exam_1000_Hz = FloatField('1000 Hz', validators=[DataRequired()])
    before_exam_2000_Hz = FloatField('2000 Hz', validators=[DataRequired()])
    before_exam_4000_Hz = FloatField('4000 Hz', validators=[DataRequired()])
    before_exam_8000_Hz = FloatField('8000 Hz', validators=[DataRequired()])
    submit = SubmitField('Predict')

@app.route('/', methods=['GET'])
def home():
    """
    Route d'accueil.
    """
    return "Welcome to the Audiometric Prediction API"

@app.route('/train', methods=['POST'])
def train():
    """
    Route pour lancer le pipeline d'entraînement du modèle.
    """
    with KedroSession.create(metadata.package_name, project_path, env="local") as session:
        context = session.load_context()
        context.run(pipeline_name="training_pipeline")
    return "Training pipeline executed successfully"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route pour effectuer des prédictions à partir de données JSON envoyées via une requête POST.
    """
    data = request.get_json()
    df = pd.DataFrame(data)
    
    # Information sur les valeurs manquantes
    missing_values_info = df.isnull().sum(axis=1)
    rows_with_missing_values = missing_values_info[missing_values_info > 0].index.tolist()
    
    # Exécution du pipeline de prédiction
    with KedroSession.create(metadata.package_name, project_path, env="local") as session:
        context = session.load_context()
        output = context.run(pipeline_name="model_prediction", data_catalog={"raw_daily_data": df})
    
    prediction = output['x_test'].to_json(orient='records')
    response = {
        "prediction": prediction,
        "missing_values_info": rows_with_missing_values
    }
    return jsonify(response)

@app.route('/form', methods=['GET', 'POST'])
def form():
    """
    Route pour afficher et traiter un formulaire HTML pour les prédictions.
    """
    form = PredictionForm()
    prediction = None
    if form.validate_on_submit():
        data = {
            'before_exam_125_Hz': [form.before_exam_125_Hz.data],
            'before_exam_250_Hz': [form.before_exam_250_Hz.data],
            'before_exam_500_Hz': [form.before_exam_500_Hz.data],
            'before_exam_1000_Hz': [form.before_exam_1000_Hz.data],
            'before_exam_2000_Hz': [form.before_exam_2000_Hz.data],
            'before_exam_4000_Hz': [form.before_exam_4000_Hz.data],
            'before_exam_8000_Hz': [form.before_exam_8000_Hz.data],
        }
        df = pd.DataFrame(data)
        
        # Exécution du pipeline de prédiction
        with KedroSession.create(metadata.package_name, project_path, env="local") as session:
            context = session.load_context()
            output = context.run(pipeline_name="model_prediction", data_catalog={"raw_daily_data": df})
        
        prediction = output['x_test'].to_json(orient='records')
    
    return render_template('index.html', form=form, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
