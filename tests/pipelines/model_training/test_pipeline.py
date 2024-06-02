# tests/pipelines/model_training/test_nodes.py

import pytest
import pandas as pd
from tonal_project.pipelines.model_training.nodes import create_model, train_model

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
        'after_exam_125_Hz': [1.0, 2.0, 3.0],
        'after_exam_250_Hz': [2.0, 3.0, 4.0],
        'after_exam_500_Hz': [3.0, 4.0, 1.0],
        'after_exam_1000_Hz': [1.0, 2.0, 3.0],
        'after_exam_2000_Hz': [2.0, 3.0, 4.0],
        'after_exam_4000_Hz': [3.0, 4.0, 1.0],
        'after_exam_8000_Hz': [1.0, 2.0, 1.0]
    }
    return pd.DataFrame(data)

def test_create_model():
    model = create_model((10, 7, 1))
    assert model is not None
    assert model.input_shape == (None, 10, 7)

def test_train_model(sample_data):
    X_train = sample_data[['before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz', 'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz']]
    y_train = sample_data[['after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz', 'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz', 'after_exam_8000_Hz']]
    X_test = X_train.copy()
    y_test = y_train.copy()
    model = train_model(X_train, y_train, X_test, y_test)
    assert model is not None
    assert len(model.layers) > 0
