import pytest
import pandas as pd
from tonal_project.pipelines.collect_transformation.nodes import transform_data, split_dataset

@pytest.fixture
def sample_data():
    data = {
        'before_exam_125_Hz': [1.0, 2.0, None],
        'before_exam_250_Hz': [2.0, 3.0, 4.0],
        'before_exam_500_Hz': [None, 3.0, 1.0],
        'before_exam_1000_Hz': [1.0, None, 1.0],
        'before_exam_2000_Hz': [2.0, 3.0, 4.0],
        'before_exam_4000_Hz': [3.0, 4.0, 1.0],
        'before_exam_8000_Hz': [1.0, 2.0, 1.0],
        'after_exam_125_Hz': [1.0, 2.0, 3.0],
        'after_exam_250_Hz': [2.0, 3.0, 4.0],
        'after_exam_500_Hz': [1.0, 2.0, 1.0],
        'after_exam_1000_Hz': [1.0, 2.0, 3.0],
        'after_exam_2000_Hz': [2.0, 3.0, 4.0],
        'after_exam_4000_Hz': [3.0, 4.0, 1.0],
        'after_exam_8000_Hz': [1.0, 2.0, 1.0]
    }
    return pd.DataFrame(data)

def test_transform_data(sample_data):
    result = transform_data(sample_data)
    assert result.isnull().sum().sum() == 0  # No missing values
    assert 'before_exam_125_Hz' not in result.columns  # Column dropped
    assert 'after_exam_125_Hz' not in result.columns  # Column dropped

def test_split_dataset(sample_data):
    transformed_data = transform_data(sample_data)
    assert not transformed_data.empty  # Assure que les données transformées ne sont pas vides
    X_train, X_test, y_train, y_test = split_dataset(transformed_data)
    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty
    assert X_train.shape[1] == 4  # 4 colonnes restantes
    assert y_train.shape[1] == 4  # 4 colonnes restantes
