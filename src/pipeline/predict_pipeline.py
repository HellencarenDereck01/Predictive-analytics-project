
import sys
import pandas as pd
import pickle
from pathlib import Path

from src.exception import CustomException


class CustomData:

    def __init__(self, studytime, absences, G1, G2):

        self.studytime = studytime
        self.absences = absences
        self.G1 = G1
        self.G2 = G2

    def to_dataframe(self):

        try:
            data = {
                "studytime": [float(self.studytime)],
                "absences": [float(self.absences)],
                "G1": [float(self.G1)],
                "G2": [float(self.G2)]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = project_root / "artifacts" / "model.pkl"

    def predict(self, input_df):

        try:
            with open(self.model_path, "rb") as file:
                model = pickle.load(file)

            prediction = model.predict(input_df)

            return float(prediction[0])

        except Exception as e:
            raise CustomException(e, sys)
