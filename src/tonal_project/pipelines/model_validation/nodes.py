# Importation des bibliothèques nécessaires
import pandas as pd
from keras import Model
from mlflow import log_metrics

def evaluate_model(model: Model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """
    Évalue le modèle sur les données de test et enregistre les métriques avec MLflow.

    Paramètres:
    model (Model): Le modèle Keras à évaluer.
    X_test (pd.DataFrame): Les caractéristiques des données de test.
    y_test (pd.DataFrame): Les cibles des données de test.

    Retourne:
    dict: Un dictionnaire contenant les métriques d'évaluation.
    """
    # Évaluation du modèle sur les données de test
    results = model.evaluate(X_test, y_test, verbose=1)
    
    # Extraction des métriques d'évaluation
    evaluation_metrics = {
        "loss": results[0],
        "accuracy": results[1]
    }
    
    # Enregistrement des métriques avec MLflow
    log_metrics(evaluation_metrics)
    
    # Retourne les métriques d'évaluation
    return evaluation_metrics
