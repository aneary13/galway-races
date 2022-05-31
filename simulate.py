import pickle
import pandas as pd

from extract import generate_full_data, generate_simulation_data
from misc import kelly

model_directory = '/Users/andy.neary/PycharmProjects/galway-races/model/'
results_directory = '/Users/andy.neary/PycharmProjects/galway-races/results/'
model_id = 'HorseyGBM_202205261418.pkl'


# Function to simulate a race week
def simulate_race_week(year=2019, pot=100, kelly_frac=0.5):
    # Generate datasets
    model_data = generate_full_data(filename='all_races05_19.csv', target_label='win', year=year)
    data = generate_simulation_data(filename='all_races05_19.csv', target_label='win', year=year)

    # Load the model and make predictions
    with open(model_directory + model_id, 'rb') as fh:
        model = pickle.load(fh)
    pred = model.predict(model_data)

    # Calculate the Kelly criteria for all horses
    data['kelly'] = kelly(data, pred)

    # Sort all races by date, time and finish
    data = data.sort_values(by=['date', 'time', 'pos'])

    # Create list of race names for the week
    race_names = list(pd.unique(data['race_name']))

    # Create empty columns to be filled in the loop below
    data['bankroll'] = 0
    data['stake'] = 0
    data['returns'] = 0

    # Generate bets for every race
    bank = pot
    for race in race_names:
        data.loc[data['race_name'] == race, 'bankroll'] = bank
        data.loc[data['race_name'] == race, 'stake'] = bank * kelly_frac * \
                                                       data.loc[data['race_name'] == race, 'kelly'] * \
                                                       (data.loc[data['race_name'] == race, 'kelly'] > 0)
        total_stake = sum(data.loc[data['race_name'] == race, 'stake'])
        data.loc[data['race_name'] == race, 'returns'] = data.loc[data['race_name'] == race, 'stake'] * \
                                                         data.loc[data['race_name'] == race, 'dec'] * \
                                                         (data.loc[data['race_name'] == race, 'pos'] == '1')
        total_returns = sum(data.loc[data['race_name'] == race, 'returns'])
        bank = bank - total_stake + total_returns

    # Drop variables no longer needed
    final = data.drop(columns=['Year', 'win', 'dec'])

    # Save results to a CSV file
    final.to_csv(results_directory + str(year) + '_results.csv')


# Run the script
if __name__ == '__main__':
    # Train a model and save
    simulate_race_week(year=2019, pot=100, kelly_frac=0.5)
