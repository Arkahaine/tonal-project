# Importation des composants nécessaires de Kedro pour créer des pipelines et des nœuds
from kedro.pipeline import Pipeline, pipeline, node
# Importation de la fonction de choix de modèle définie dans nodes.py
from .nodes import choose_best_model

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée et retourne un pipeline Kedro pour la sélection du meilleur modèle.

    Paramètres:
    kwargs: Arguments supplémentaires passés à la fonction de création du pipeline.

    Retourne:
    Pipeline: Un objet pipeline de Kedro contenant le nœud de sélection du meilleur modèle.
    """
    return pipeline([
        node(
            func=choose_best_model,
            inputs=["x_test", "y_test", "model_trained", "params:model_final_path"],
            outputs="model_final",
            name="node_choose_best_model"
        )
    ])
