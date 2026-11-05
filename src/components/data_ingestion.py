import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


class DataIngestion:

    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Student Performance data ingestion")

            # Step 1: Load dataset
            df = pd.read_csv("data/student-mat.csv", sep=";")

            print("Dataset loaded successfully!")
            print("Dataset shape:", df.shape)

            # Step 2: Check required columns
            required_columns = [
                "studytime", "absences", "G1", "G2", "G3"
            ]

            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing column: {column}")

            # Step 3: Create artifacts directory
            os.makedirs("artifacts", exist_ok=True)

            # Step 4: Save raw dataset
            df.to_csv(self.raw_data_path, index=False)

            # Step 5: Split dataset
            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42
            )

            # Step 6: Save training and testing datasets
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys) from e