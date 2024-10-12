from models.Diabetes_Readmission_Prediction.model import DiabetesModel

model_path = 'models\Diabetes_Readmission_Prediction\saved_models\rf_model_selected.joblib'

def make_prediction(input_data: dict):
    model = DiabetesModel(model_path)  # Initialize the DiabetesModel object
    return model.predict(input_data)   # Call the predict method with input data


