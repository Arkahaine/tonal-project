# Importation des composants nécessaires de Kedro pour créer des pipelines et des nœuds
from kedro.pipeline import Pipeline, pipeline, node
# Importation des fonctions de transformation et de prédiction des données définies dans nodes.py
from .nodes import transform_data, predict_model

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée et retourne un pipeline Kedro pour la transformation des données et la prédiction du modèle.

    Paramètres:
    kwargs: Arguments supplémentaires passés à la fonction de création du pipeline.

    Retourne:
    Pipeline: Un objet pipeline de Kedro contenant les nœuds de transformation des données et de prédiction du modèle.
    """
    return pipeline([
        node(
            func=transform_data,
            inputs="raw_data_to_predict",
            outputs="data_to_predict",
            name="node_transform_data_to_predict"
        ),
        node(
            func=predict_model,
            inputs=["data_to_predict", "model_trained"],
            outputs=None,  # Les prédictions ne sont pas stockées dans une variable de sortie
            name="node_predict_model"
        )
    ])
