import logging
import pickle
import pandas as pd

from extract import generate_split_data
from model import HorseyGBM
from misc import metrics

logging.basicConfig(level=logging.INFO)


# Define a function to evaluate and log results
def log_performance(predictions, split, first_val_year, first_test_year, target_label):

    # Define metric names
    names = ['% of placed bets that won', '% of winning horses that were bet on']

    # Calculate performance metrics
    values = metrics(predictions, split, first_val_year, first_test_year, target_label)

    # Log the performance
    for name, value in zip(names, values):
        logging.info('{} set - {} : {:.2f}'.format(split.capitalize(), name, value))


# Function to get a model and fit it on some data
def get_fitted_model(target_label='win', first_val_year=2018, first_test_year=2019):

    # Generate training, validation and test sets
    train, val, test = generate_split_data(first_val_year=first_val_year, first_test_year=first_test_year)

    # Create the model and fit on the training data
    model = HorseyGBM(target_label=target_label)
    model.fit(train, val)

    # Make predictions on the train, validation and test sets
    train_pred = model.predict(train)
    val_pred = model.predict(val)
    test_pred = model.predict(test)

    # Log model performance
    log_performance(train_pred, 'train', first_val_year, first_test_year, target_label)
    log_performance(val_pred, 'val', first_val_year, first_test_year, target_label)
    log_performance(test_pred, 'test', first_val_year, first_test_year, target_label)

    # Save the trained model
    model_id = pd.to_datetime('now', utc=True).strftime('%Y%m%d%H%M')
    with open('model/HorseyGBM_{}.pkl'.format(model_id), 'wb') as fh:
        pickle.dump(model, fh)


# Run the script
if __name__ == '__main__':

    # Train a model and save
    get_fitted_model(target_label='win', first_val_year=2018, first_test_year=2019)
