# Importation des bibliothèques nécessaires
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

def choose_best_model(X_test, y_test, new_model, old_model_path):
    """
    Choisit le meilleur modèle entre un nouveau modèle et un modèle existant basé sur les performances de test.

    Paramètres:
    X_test (pd.DataFrame): Les données de test pour les caractéristiques.
    y_test (pd.Series): Les données de test pour les cibles.
    new_model (sklearn model): Le nouveau modèle à évaluer.
    old_model_path (str): Le chemin vers le fichier pickle du modèle existant.

    Retourne:
    sklearn model: Le meilleur modèle basé sur les erreurs moyennes absolues et quadratiques.
    """
    try:
        # Tentative de chargement du modèle existant à partir d'un fichier pickle
        with open(old_model_path, 'rb') as f:
            old_model = pickle.load(f)
    except FileNotFoundError:
        # Si le fichier du modèle existant n'est pas trouvé, retourner le nouveau modèle
        return new_model

    # Prédictions avec le nouveau modèle
    y_pred = new_model.predict(X_test)
    # Calcul des erreurs pour le nouveau modèle
    new_mae = mean_absolute_error(y_test, y_pred)
    new_mse = mean_squared_error(y_test, y_pred)

    # Prédictions avec le modèle existant
    y_pred_old = old_model.predict(X_test)
    # Calcul des erreurs pour le modèle existant
    old_mae = mean_absolute_error(y_test, y_pred_old)
    old_mse = mean_squared_error(y_test, y_pred_old)

    # Comparaison des modèles basé sur les erreurs moyennes quadratiques et absolues
    if new_mse > old_mse or new_mae > old_mae:
        # Si le nouveau modèle est moins performant, retourner l'ancien modèle
        return old_model
    else:
        # Si le nouveau modèle est plus performant ou égal, retourner le nouveau modèle
        return new_model
