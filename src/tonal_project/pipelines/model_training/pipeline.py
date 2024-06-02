# Importation des composants nécessaires de Kedro pour créer des pipelines et des nœuds
from kedro.pipeline import Pipeline, pipeline, node
# Importation de la fonction d'entraînement de modèle définie dans nodes.py
from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée et retourne un pipeline Kedro pour l'entraînement du modèle.

    Paramètres:
    kwargs: Arguments supplémentaires passés à la fonction de création du pipeline.

    Retourne:
    Pipeline: Un objet pipeline de Kedro contenant le nœud d'entraînement du modèle.
    """
    return pipeline([
        node(
            func=train_model,
            inputs=["x_train", "y_train", "x_test", "y_test"],
            outputs="model_trained",
            name="node_train_model"
        )
    ])
