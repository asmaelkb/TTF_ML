import mlflow
import mlflow.pyfunc
import pandas as pd

def make_predictions():
    new_data = pd.read_csv('new_data.csv')
    
    # Load the model from MLflow
    model_uri = "models:/TimeToFailureModel/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    predictions = model.predict(new_data)
    
    # Save the predictions as artifacts in MLflow
    with mlflow.start_run() as run:
        predictions_df = pd.DataFrame(predictions, columns=['prediction'])
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")

if __name__ == "__main__":
    make_predictions()
