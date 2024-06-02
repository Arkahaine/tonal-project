# Importation des composants nécessaires de Kedro pour créer des pipelines et des nœuds
from kedro.pipeline import Pipeline, pipeline, node
# Importation des fonctions de transformation de données définies dans nodes.py
from .nodes import transform_data, split_dataset

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée et retourne un pipeline Kedro pour la transformation et la séparation des données.

    Paramètres:
    kwargs: Arguments supplémentaires passés à la fonction de création du pipeline.

    Retourne:
    Pipeline: Un objet pipeline de Kedro contenant les nœuds de transformation et de séparation des données.
    """
    return pipeline([
        node(
            func=transform_data,
            inputs="raw_daily_data",
            outputs="shaped_datas",
            name="node_merge_raw_daily_data"
        ),
        node(
            func=split_dataset,
            inputs="shaped_datas",
            outputs=["x_train", "x_test", "y_train", "y_test"],
            name="node_split_transform_daily_data"
        )
    ])
