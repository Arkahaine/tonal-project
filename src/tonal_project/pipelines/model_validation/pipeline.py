# Importation des composants nécessaires de Kedro pour créer des pipelines et des nœuds
from kedro.pipeline import Pipeline, pipeline, node
# Importation de la fonction d'évaluation du modèle définie dans nodes.py
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée et retourne un pipeline Kedro pour l'évaluation du modèle.

    Paramètres:
    kwargs: Arguments supplémentaires passés à la fonction de création du pipeline.

    Retourne:
    Pipeline: Un objet pipeline de Kedro contenant le nœud d'évaluation du modèle.
    """
    return pipeline([
        node(
            func=evaluate_model,
            inputs=["model_trained", "x_test", "y_test"],
            outputs=None,  # Les résultats de l'évaluation ne sont pas stockés dans une variable de sortie
            name="evaluate_model_node"
        )
    ])
