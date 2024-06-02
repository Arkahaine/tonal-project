# Importation des bibliothèques nécessaires
from keras import Model
import pandas as pd

def transform_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données en s'assurant que toutes les valeurs sont des flottants et en supprimant les lignes contenant des valeurs nulles.
    
    Paramètres:
    input_data (pd.DataFrame): Le DataFrame d'entrée à transformer.
    
    Retourne:
    pd.DataFrame: Le DataFrame transformé avec les colonnes spécifiées supprimées et les valeurs nulles éliminées.
    """
    # Conversion de toutes les valeurs en type float, remplacement des erreurs par NaN
    data_transformed = input_data.apply(pd.to_numeric, errors='coerce')
    
    # Suppression des lignes contenant des valeurs nulles
    data_transformed.dropna(inplace=True)

    # Liste des colonnes à supprimer
    columns_to_drop = [
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_8000_Hz', 'after_exam_125_Hz', 'after_exam_250_Hz',
        'after_exam_500_Hz', 'after_exam_8000_Hz'
    ]
    
    # Suppression des colonnes spécifiées
    data_transformed.drop(columns=columns_to_drop, inplace=True)

    # Retourne le DataFrame transformé
    return data_transformed

def predict_model(input_data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """
    Prédit les résultats du modèle en utilisant les données d'entrée.
    
    Paramètres:
    input_data (pd.DataFrame): Les données d'entrée pour la prédiction.
    model (Model): Le modèle Keras à utiliser pour les prédictions.
    
    Retourne:
    pd.DataFrame: Les résultats prédits par le modèle.
    """
    # Prédiction des résultats en utilisant le modèle
    data_predicted = model.predict(input_data)

    # Conversion des prédictions en DataFrame pour un retour structuré
    data_predicted = pd.DataFrame(data_predicted, index=input_data.index, columns=["prediction"])

    # Retourne les données prédites
    return data_predicted
