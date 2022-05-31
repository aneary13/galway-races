import pandas as pd

default_file_path = '/Users/andy.neary/Downloads/'
model_features = [
    'Year',
    'class',
    'band',
    'dist.m.',
    'going',
    'race_group',
    'race_type',
    'Runners',
    'Race_Money',
    'trainer',
    'jockey',
    'age',
    'lbs',
    'gear',
    'ts',
    'or',
    'rpr',
    'dec',
    'pos'
]
simulation_features = [
    'Year',
    'date',
    'time',
    'race_name',
    'horse_name',
    'sp',
    'dec',
    'pos',
]


# Function to generate a dataset from a CSV file
def create_dataset(filename='all_races05_19.csv', target_label='win', sim=False):

    # Load the CSV file
    df = pd.read_csv(default_file_path + filename, low_memory=False)

    # Select relevant features for Galway races only
    if sim:
        data = df.loc[df['course'] == 'Galway', simulation_features]
    else:
        data = df.loc[df['course'] == 'Galway', model_features]

    # Create variables for win and place
    data['win'] = (data['pos'] == '1') * 1
    data['place'] = (data['pos'].isin(['1', '2', '3', '4'])) * 1
    if sim:
        pass
    else:
        data = data.drop('pos', axis=1)

    # Drop the target label not being used
    if target_label not in ['win', 'place']:
        raise ValueError("Make sure target_value is set to either 'win' or 'place'")

    # Drop 'place' if the target is 'win'
    elif target_label == 'win':
        dataset = data.drop('place', axis=1)

    # Drop 'win' if the target is 'place'
    else:
        dataset = data.drop('win', axis=1)

    # Return the generated dataset
    return dataset


# Function to create train, validation and test splits
def generate_split_data(filename='all_races05_19.csv', target_label='win', first_val_year=2018, first_test_year=2019):

    # Create a dataset
    df = create_dataset(filename, target_label)

    # Create the boolean masks
    train_mask = df['Year'] < first_val_year
    val_mask = (df['Year'] >= first_val_year) & (df['Year'] < first_test_year)
    test_mask = df['Year'] >= first_test_year

    # Split the dataset
    train = df.loc[train_mask, :].drop(columns=['Year'])
    val = df.loc[val_mask, :].drop(columns=['Year'])
    test = df.loc[test_mask, :].drop(columns=['Year'])

    # Return the split sets
    return train, val, test


# Function to create train, validation and test splits
def generate_full_data(filename='all_races05_19.csv', target_label='win', year=2019):

    # Create a dataset
    df = create_dataset(filename, target_label)

    # Select data for year of interest only
    data = df.loc[df['Year'] == year, :]

    # Drop the year
    dataset = data.drop('Year', axis=1)

    # Return the split sets
    return dataset


# Function to generate data used for calculating performance metrics
def generate_performance_data(filename='all_races05_19.csv', target_label='win', start=None, end=None):

    # Create a dataset
    df = create_dataset(filename, target_label)

    # Select features used for calculating performance metrics
    data = df.loc[:, ['Year', 'dec', target_label]]

    # Create mask for year(s) of interest
    start = 2005 if not start else start
    end = 2020 if not end else end
    mask = (data['Year'] >= start) & (data['Year'] < end)
    dataset = data.loc[mask, :]

    # Return the dataset
    return dataset


# Function to generate data used for calculating performance metrics
def generate_simulation_data(filename='all_races05_19.csv', target_label='win', year=2019):

    # Create a dataset
    df = create_dataset(filename, target_label, sim=True)

    # Select data for year of interest only
    dataset = df.loc[df['Year'] == year, :]

    # Return the dataset
    return dataset
