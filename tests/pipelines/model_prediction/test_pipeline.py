import pytest
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tonal_project.pipelines.model_prediction.nodes import transform_data, predict_model

@pytest.fixture
def sample_data():
    data = {
        'before_exam_125_Hz': [1.0, 2.0, 3.0],
        'before_exam_250_Hz': [2.0, 3.0, 4.0],
        'before_exam_500_Hz': [3.0, 4.0, 1.0],
        'before_exam_1000_Hz': [1.0, 2.0, 3.0],
        'before_exam_2000_Hz': [2.0, 3.0, 4.0],
        'before_exam_4000_Hz': [3.0, 4.0, 1.0],
        'before_exam_8000_Hz': [1.0, 2.0, 3.0],
        'after_exam_125_Hz': [1.0, 2.0, 3.0],  # Ajout des colonnes manquantes
        'after_exam_250_Hz': [2.0, 3.0, 4.0],
        'after_exam_500_Hz': [1.0, 2.0, 1.0],
        'after_exam_1000_Hz': [1.0, 2.0, 3.0],
        'after_exam_2000_Hz': [2.0, 3.0, 4.0],
        'after_exam_4000_Hz': [3.0, 4.0, 1.0],
        'after_exam_8000_Hz': [1.0, 2.0, 1.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(7,)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def test_transform_data(sample_data):
    result = transform_data(sample_data)
    assert result.isnull().sum().sum() == 0  # No missing values

def test_predict_model(sample_data, dummy_model):
    transformed_data = transform_data(sample_data)
    predictions = predict_model(transformed_data, dummy_model)
    assert len(predictions) == len(transformed_data)
