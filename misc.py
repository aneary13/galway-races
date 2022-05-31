from extract import generate_performance_data


# Function to calculate Kelly criterion
def kelly(data, predictions):

    # Add predictions to data
    data.loc[:, 'pred'] = predictions

    # Calculate Kelly criterion
    b = data['dec'] - 1
    p = data['pred']
    q = 1 - p
    k = ((b * p) - q) / b

    # Return Kelly criteria
    return k


# Function to calculate performance metrics
def metrics(predictions, split, first_val_year, first_test_year, target_label):

    # Define start and end of period based on split
    if split not in ['train', 'val', 'test']:
        raise ValueError("Make sure split i either 'train', 'val' or 'test'")

    elif split == 'train':
        start = None
        end = first_val_year

    elif split == 'val':
        start = first_val_year
        end = first_test_year

    else:
        start = first_test_year
        end = None

    # Create dataset
    df = generate_performance_data(filename='all_races05_19.csv', target_label=target_label, start=start, end=end)

    # Make a copy of the DataFrame
    df = df.copy()

    # Add predictions to data
    df.loc[:, 'kelly'] = kelly(df, predictions)

    # Calculate performance metrics
    num_bets_made = sum(df['kelly'] > 0)
    num_winning_horses = sum(df[target_label] == 1)
    num_winning_bets = sum((df['kelly'] > 0) & (df[target_label] == 1))
    pct_winning_bets = 100 * num_winning_bets / num_bets_made
    pct_winners_found = 100 * num_winning_bets / num_winning_horses

    # Return metrics
    return pct_winning_bets, pct_winners_found
