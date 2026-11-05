
import os
import sys
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException


class DataTransformation:

    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts", "preprocessor.pkl"
        )

    def get_data_transformer(self, numeric_cols):
        try:
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols)
                ],
                remainder="drop"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            features = [
                "studytime",
                "absences",
                "G1",
                "G2"
            ]

            target = "G3"

            X_train = train_df[features]
            y_train = train_df[target]

            X_test = test_df[features]
            y_test = test_df[target]

            preprocessor = self.get_data_transformer(features)

            # Learn preprocessing rules from training data only
            X_train_transformed = preprocessor.fit_transform(X_train)

            # Apply the learned rules to test data
            X_test_transformed = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as file:
                pickle.dump(preprocessor, file)

            logging.info("Preprocessor saved successfully")

            logging.info("Data transformation completed")

            return (
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys) from e
