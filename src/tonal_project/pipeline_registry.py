# Importation des bibliothèques nécessaires
from typing import Dict
from kedro.pipeline import Pipeline

# Importation des modules de pipelines spécifiques au projet
from tonal_project.pipelines import (
    collect_transformation,
    model_training,
    model_validation,
    model_deployment,
    model_prediction
)

def register_pipelines() -> Dict[str, Pipeline]:
    """
    Enregistre les pipelines du projet.

    Retourne:
        Un dictionnaire mappant les noms des pipelines aux objets ``Pipeline`` correspondants.
    """
    # Création des instances de chaque pipeline en appelant leur fonction de création respective
    collect_transformation_pipeline = collect_transformation.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_validation_pipeline = model_validation.create_pipeline()
    model_deployment_pipeline = model_deployment.create_pipeline()
    model_prediction_pipeline = model_prediction.create_pipeline()

    # Retourne un dictionnaire qui mappe les noms de pipelines à leurs objets Pipeline respectifs
    return {
        "collect_transformation": collect_transformation_pipeline,
        "model_training": model_training_pipeline,
        "model_validation": model_validation_pipeline,
        "model_deployment": model_deployment_pipeline,
        "model_prediction": model_prediction_pipeline,
        "__default__": collect_transformation_pipeline + model_training_pipeline + model_validation_pipeline + model_deployment_pipeline
    }
