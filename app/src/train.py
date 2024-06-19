import pandas as pd
import autokeras as ak
import datetime
from preprocess import preprocess_data
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def input_df(df, tags, start_date, end_date): 
    """ Returns the dataframe with the given list of tags, between the start_date and the end_date. """
    df_index = pd.to_datetime(df.index)

    df_index_filtered = df_index[(df_index >= start_date) & (df_index <= end_date)].drop_duplicates()
    data = df.loc[df_index_filtered]

    return data

def input_countdown(df, start_date, end_date):
    """ Returns the Countdown dataframe between the start_date and the end_date. """
    df_index = pd.to_datetime(df.index)

    # df_index_filtered = df_index[(df_index >= start_date) & (df_index <= end_date)]
    mask = (df_index >= start_date) & (df_index <= end_date)
    data = df.loc[mask]

    data = data.loc[data.index.repeat(10)]
    
    return data.index, data['Countdown']

class CustomAutoModel(ak.AutoModel):
    def compile(self, **kwargs):
        # Call the original compile method with custom arguments
        super().compile(metrics=['accuracy'], **kwargs)

def split(df_cgce, df_countdown, tags):
    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-05-30 00:00:00')
    train_data = input_df(df_cgce, tags, start_date, end_date)

    X_train = train_data.to_numpy()
    y_train_index, y_train = input_countdown(df_countdown, start_date, end_date)
    y_train = y_train.to_numpy()

    start_date = pd.to_datetime('2019-05-30 01:00:00') 
    end_date = pd.to_datetime('2019-06-30 14:00:00') 
    test_data = input_df(df_cgce, tags, start_date, end_date)

    test_data = test_data[test_data.index.minute == 0]
    test_data = test_data.head(750)

    X_test = test_data.to_numpy()
    y_test_index, y_test = input_countdown(df_countdown, start_date, end_date)
    
    y_test_index = y_test_index[y_test_index.minute == 0]
    y_test_index = y_test_index[:750]

    y_test = y_test[y_test.index.minute == 0]
    y_test = y_test.head(750)

    y_test = y_test.to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model():

    df_countdown = pd.read_csv("/raid/intern_asmae/TTF_ML/app/src/results/countdown_all.csv", parse_dates=['Time'])
    df_countdown = df_countdown.set_index('Time')

    df_corr = pd.read_csv("/raid/intern_asmae/TTF_ML/app/src/results/correlation.csv")

    # Using the first 10 tags (the ones with the highest correlation with countdown)
    tags = df_corr['Tag'].head(10)
    # tags50 = df_corr['Tag'].head(50)

    df_cgce = preprocess_data(tags) # modifier preprocess_data pour renvoyer également df_countdown et tags
    

    X_train, X_val, X_test, y_train, y_val, y_test = split(df_cgce, df_countdown, tags)

    input_node = ak.Input()
    output_node = ak.RegressionHead()(input_node)
    regressor = CustomAutoModel(
        inputs=input_node,
        outputs=output_node,
        max_trials=10,  # Number of tests of different architectures
        overwrite=True,
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    mlflow.set_tracking_uri('http://172.16.239.131:5000')
    mlflow.set_experiment(experiment_name="first")

    with mlflow.start_run():
        # Early stopping : stop before the validation loss increases
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        regressor.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[tensorboard_callback, early_stopping], verbose=2)

        # Get the best trials
        best_trials = regressor.tuner.oracle.get_best_trials(num_trials=1)
        best_trial = best_trials[0]

        # Print the best trial details
        print(f"Best trial ID: {best_trial.trial_id}")

        # Evaluation of the model
        best_model = regressor.export_model()

        best_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        y_test_pred = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        mlflow.log_metric("mse", test_mse)
        mlflow.log_metric("mae", test_mae)

        # Save the best model
        best_model.save("trained_model.keras")
        print("Model saved successfully in trained_model.keras")

        mlflow.sklearn.log_model(best_model, "model")

if __name__ == "__main__":
    train_model()