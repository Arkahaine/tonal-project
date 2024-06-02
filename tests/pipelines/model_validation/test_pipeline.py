# tests/pipelines/model_validation/test_nodes.py

import pandas as pd
import pytest
from keras.models import Sequential
from keras.layers import Dense
from tonal_project.pipelines.model_validation.nodes import evaluate_model

@pytest.fixture
def sample_data():
    data = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [2.0, 3.0, 4.0],
        'target': [1.0, 2.0, 3.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(2,)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def test_evaluate_model(sample_data, dummy_model):
    X_test = sample_data[['feature1', 'feature2']]
    y_test = sample_data[['target']]
    dummy_model.fit(X_test, y_test, epochs=1, verbose=0)
    metrics = evaluate_model(dummy_model, X_test, y_test)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
