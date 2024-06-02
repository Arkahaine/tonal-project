# Importation des bibliothèques nécessaires
from keras import layers, regularizers, optimizers, metrics, Model, callbacks
import pandas as pd
import mlflow

# Activation de l'enregistrement automatique des expériences avec MLflow
mlflow.autolog()

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=0.5, learning_rate=1e-3):
    """
    Crée et compile un modèle Keras basé sur une architecture Conv1D.

    Paramètres:
    input_shape (tuple): La forme des données d'entrée, doit avoir trois dimensions.
    units (int): Le nombre de neurones dans la couche dense.
    activation (str): La fonction d'activation à utiliser dans les couches cachées.
    l2_value (float): La valeur de régularisation L2.
    dropout_rate (float): Le taux de dropout pour la couche Dropout.
    learning_rate (float): Le taux d'apprentissage pour l'optimiseur Adam.

    Retourne:
    Model: Le modèle Keras compilé.
    """
    print(f"Creating model with input shape: {input_shape}")

    # Vérification que la forme d'entrée a bien trois dimensions
    if len(input_shape) != 3:
        raise ValueError(f"Expected input shape to have 3 dimensions, got {len(input_shape)} dimensions.")

    # Définition de l'input du modèle
    inputs = layers.Input(shape=(input_shape[1], input_shape[2]))

    # Ajout de couches Conv1D et de pooling
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation, padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation=activation, padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Aplatissement des résultats et ajout d'une couche dense
    x = layers.Flatten()(x)
    x = layers.Dense(units, activation=activation, kernel_regularizer=regularizers.L2(l2_value))(x)

    # Ajout de la couche de dropout si le taux de dropout est spécifié
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # Définition de la couche de sortie
    outputs = layers.Dense(input_shape[1], activation='linear')(x)

    # Création du modèle
    model = Model(inputs=inputs, outputs=outputs)

    # Compilation du modèle
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanSquaredError()])

    return model

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Entraîne un modèle Keras sur les données d'entraînement et de validation.

    Paramètres:
    X_train (pd.DataFrame): Les caractéristiques des données d'entraînement.
    y_train (pd.DataFrame): Les cibles des données d'entraînement.
    X_test (pd.DataFrame): Les caractéristiques des données de test.
    y_test (pd.DataFrame): Les cibles des données de test.

    Retourne:
    Model: Le modèle Keras entraîné.
    """
    # Reshape des données pour qu'elles soient compatibles avec Conv1D
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_train_reshaped = y_train.values.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_test_reshaped = y_test.values.reshape((y_test.shape[0], y_test.shape[1], 1))

    # Affichage des nouvelles formes des données pour vérification
    print(f"X_train reshaped shape: {X_train_reshaped.shape}")
    print(f"X_test reshaped shape: {X_test_reshaped.shape}")
    print(f"y_train reshaped shape: {y_train_reshaped.shape}")
    print(f"y_test reshaped shape: {y_test_reshaped.shape}")

    # Création du modèle avec les paramètres spécifiés
    model = create_model(X_train_reshaped.shape, dropout_rate=0.5)

    # Définition du callback d'arrêt anticipé pour éviter le sur-apprentissage
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entraînement du modèle avec les données d'entraînement et de validation
    history = model.fit(X_train_reshaped, y_train_reshaped, epochs=10, validation_data=(X_test_reshaped, y_test_reshaped), callbacks=[early_stopping])

    # Retourne le modèle entraîné
    return model  # Return only the model
