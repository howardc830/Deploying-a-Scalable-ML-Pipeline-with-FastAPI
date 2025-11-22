import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    performance_on_categorical_slice
)
from ml.data import process_data

@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv("data/census.csv")
    return df.sample(n=200, random_state=28)
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_random_forest_classifier(sample_data):
   """Test that train_model returns a RandomForestClassifier instance."""
    train_df, _ = train_test_split(
        sample_data, test_size=0.2, random_state=42, stratify=sample_data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics_returns_values_between_0_and_1(sample_data):
    """
    # Test that precision, recall, and fbeta are floats between 0 and 1
    """
    train, test = train_test_split(sample_data, test_size=0.2, random_state=28 stratify=sample_data["salary"])

    X_train, y_train, encoder, lb = process_data(
         train,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"],
        label="salary", training=False, encoder=encoder, lb=lb)

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"],
        label="salary", training=False, encoder=encoder, lb=lb)

model = train_model(X_train, y_train)
preds = inference(model, X_test)
precsion, recall, fbeta = compute_model_metrics(y_test, preds)

assert isinstance(precision, float)
assert isinstance(recall, float)
assert isinstance(fbeta, float)
assert 0.0 <= precision <= 1.0
assert 0.0 <= recall <= 1.0
assert 0.0 <= fbeta <= 1.0


# TODO: implement the third test. Change the function name and input as needed
def test_inference_returns_correct_shape_and_type(sample_data):
    """
    # Test that inference returns a numpy array with correct length and integer datatype
    """
    train, test = train_test_split(sample_data, test_size=0.2, random_state=28, stratify=sample_data["salary"])
    X_train, y_train, encoder, lb = process_data(
        train,
        ategorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"],
    )
        X_test, y_test, _, _ = process_data(
        test,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"],
        label="salary", training=False, encoder=encoder, lb=lb)

model = train_model(X_train, y_train)
predictions = inference(model, X_test)

assert isInstance(predictions, np.ndarray), "Predictions should be a numpy array"
assert len(predictions) == len(X_test), "Prediction count doesn't match test set size"
assert np.issubdtype(predictions.dtype, np.integer), "Predictions should be integer type"
        
