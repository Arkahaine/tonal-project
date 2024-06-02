# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split

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

def split_dataset(input_data: pd.DataFrame):
    """
    Sépare le jeu de données en ensembles d'entraînement et de test pour les prédicteurs (X) et les cibles (y).
    
    Paramètres:
    input_data (pd.DataFrame): Le DataFrame d'entrée à diviser.
    
    Retourne:
    tuple: Tuple contenant les ensembles d'entraînement et de test pour X et y.
    """
    # Sélection des colonnes prédicteurs (commençant par 'before_') et cibles (commençant par 'after_')
    X = input_data.filter(regex='^before_')
    y = input_data.filter(regex='^after_')

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création de DataFrames pour les ensembles d'entraînement et de test avec les mêmes colonnes que les données originales
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    y_train = pd.DataFrame(y_train, columns=y.columns)
    y_test = pd.DataFrame(y_test, columns=y.columns)

    # Affichage des formes des ensembles d'entraînement et de test
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Retourne les ensembles d'entraînement et de test pour X et y
    return X_train, X_test, y_train, y_test
