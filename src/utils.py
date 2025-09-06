import os
import dill
import sys
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save a Python object to disk using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        print(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Train multiple models and return their R2 score on test data.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
