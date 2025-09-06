import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models)
            best_score = max(report.values())
            best_model_name = list(report.keys())[list(report.values()).index(best_score)]
            best_model = models[best_model_name]

            save_object(self.config.trained_model_file_path, best_model)

            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            logging.info(f"Best model: {best_model_name}, R2 Score: {r2}")
            return best_model_name, r2

        except Exception as e:
            raise CustomException(e, sys)
