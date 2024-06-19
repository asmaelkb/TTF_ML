import pandas as pd
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from tensorflow.keras.callbacks import TensorBoard
import datetime

def input_df(df, tags, start_date, end_date): 
    """ Returns the dataframe with the given list of tags, between the start_date and the end_date. """
    df_index = pd.to_datetime(df.index)

    df_index_filtered = df_index[(df_index >= start_date) & (df_index <= end_date)]
    data = df.loc[df_index_filtered]

    return data[tags]

def input_countdown(df, start_date, end_date):
    """ Returns the Countdown dataframe between the start_date and the end_date. """
    df_index = pd.to_datetime(df.index)

    # df_index_filtered = df_index[(df_index >= start_date) & (df_index <= end_date)]
    mask = (df_index >= start_date) & (df_index <= end_date)
    data = df.loc[mask]
    
    return data['Countdown']

class Model():
    """ This class defines the architecture of our LSTM model and allows you to train it"""
    
    def __init__(self, input_shape):
        inputs = Input(shape=input_shape)
        self.model = Sequential()
        self.model.add(inputs)
        self.model.add(LSTM(50))
        self.model.add(Dense(1, activation='sigmoid'))  # Output layer (1 output only)

    def train(self, X_train, Y_train, epochs=10, callbacks=None):
        """ Trains the model on the provided training data """
        # callbacks = callbacks or [] 
        # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # callbacks = [early_stopping] + callbacks
        self.model.fit(X_train, Y_train, epochs=epochs, callbacks=callbacks)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        """ Defining the training parameters such as the optimizer and the loss function"""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def predict(self, X):
        """ Returns the prediction of the model for a given data input """
        return self.model(X)

class ModelWrapper(BaseEstimator):
    """Wrapper class for the Model class to conform with scikit-learn's estimator interface"""

    def __init__(self, input_shape, epochs=10, callbacks=None):
        self.input_shape = input_shape
        self.epochs = epochs
        self.callbacks = callbacks

    def fit(self, X, y):
        # Instantiate your Model class
        self.model = Model(self.input_shape)
        self.model.compile()

        # Train the model
        self.model.train(X, y, epochs=self.epochs, callbacks=self.callbacks)

    def predict(self, X):
        # Return predictions using the trained model
        return self.model.predict(X)

def custom_scorer(model, X, y):
    _, accuracy = model.evaluate(X, y, verbose=0)
    return accuracy  

# Define a custom scorer function for RandomizedSearchCV
def keras_scorer(estimator, X, y):
    return custom_scorer(estimator.model.model, X, y)



if __name__ == '__main__':

    # Loading the dataframes
    df_cgce = pd.read_csv('data/CGCE7K-1920-0503-Pivot15M-updated.csv', parse_dates=['Time'])
    df_cgce = df_cgce.set_index('Time')

    df_countdown = pd.read_csv("src/results/countdown_all.csv")
    df_countdown = df_countdown.set_index('Time')

    df_corr = pd.read_csv("src/results/correlation.csv")

    # Using the first 10 tags (the ones with the highest correlation with countdown)
    tags = df_corr['Tag'].head(10)

    # Train/Test splitting 
    start_date = pd.to_datetime('2019-05-01 00:00:00')
    end_date = pd.to_datetime('2019-05-30 00:00:00')
    train_data = input_df(df_cgce, tags, start_date, end_date)

    X_train = train_data.to_numpy()
    y_train = input_countdown(df_countdown, start_date, end_date).to_numpy()

    start_date = pd.to_datetime('2019-05-30 01:00:00') 
    end_date = pd.to_datetime('2019-06-30 14:00:00') 
    test_data = input_df(df_cgce, tags, start_date, end_date)

    X_test = test_data.to_numpy()
    y_test = input_countdown(df_countdown, start_date, end_date).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Remove nan values
    imputer = SimpleImputer(strategy='mean')  # Unique nan value to impute
    X_train = imputer.fit_transform(X_train)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    
    # Define the model
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Training
    param_dist = {
        'input_shape': [(X_train.shape[1], X_train.shape[2])],  # Provide a list of input shapes
        'epochs': [20],  # Number of epochs for training
    }

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=ModelWrapper(input_shape, callbacks=[tensorboard_callback]), param_distributions=param_dist, cv=5, n_iter=100, n_jobs=-1, scoring=keras_scorer)
    random_search.fit(X_train, y_train)

    # Print best parameters and best score
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)

    # Get the best model
    best_model = random_search.best_estimator_

    # Evaluation
    loss, accuracy = best_model.model.model.evaluate(X_val, y_val)
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)

    # Save the model
    best_model.model.model.save("trained_model.keras")
    print("Model saved successfully in trained_model.keras")
