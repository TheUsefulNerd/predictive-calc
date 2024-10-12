import pandas as pd
import joblib

class DiabetesModel:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Perform necessary preprocessing steps, e.g., encoding, scaling
        # Example: Convert categorical variables, handle missing values, etc.
        age_bins = {'10-20': 0, '20-30': 1, '30-40': 2, '40-50': 3, '50-60': 4, '60-70': 5, '70-80': 6, '80+': 7}
        data['age'] = data['age'].map(age_bins)
        return data

    def predict(self, input_data: dict) -> int:
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])
        processed_data = self.preprocess(input_df)
        prediction = self.model.predict(processed_data)
        return prediction[0]


