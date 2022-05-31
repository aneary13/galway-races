# Galway Races Money Printing Extravaganza Machine

## The Idea

I'm trying to develop a model that goes through the following steps:
1. Take as input a CSV file with data relating to upcoming horse races.
2. Use a machine learning model to predict the probability of a horse winning a given race.
3. Use fractional Kelly betting to decide whether to bet on a horse, and if so how much.

## The Data

The dataset 'Horse Racing Results - UK & Ireland 2005 to 2019' from Kaggle is used for developing and testing the model. I only use Galway races for training and testing.

Some thoughts:
- Should I be using more than just Galway data?
- I noticed when looking at results that not all races are present for the Galway 2019 summer races. Check up on this.
- Once initial development is complete, I plan to manually transcribe results from 2020 and 2021 to test on these years as well. This will also give me a better sense of how to go about things when it comes to the 2022 races.

## Data Pre-Processing

In order to develop a machine learning model, the data needs some pre-processing first. I go through the following steps:
- Remove irrelevant features.
- Create a binary variable describing if the horse won the race or not.
- Fill missing categorical data with 'Unknown'.
- Impute missing numerical data.
- Generate a list of top trainers, and fill replace less successful trainers with 'Other'.
- Repeat the above step for jockeys.
- Create new features for all features with missing data, to flag which horses had missing data.

Some thoughts:
- Imputing 'or' (Official Rating) may not be the best way to go, it might be better to fill with zeros. This is because all horses will have an Official Rating unless they are maidens (I think).
- Treating 'class' as a categorical variable may not be the way to go. A lot of classes seem to specify an Official Rating range which horses must fall in to qualify for a race. It might be better to split 'class' into two features describing the minimum and maximum qualifying Official Rating. I need to learn more about what class means before going about that. One problem this would solve is that more recent races (e.g. 2021) seem to use highly overlapping but slightly different classes to those present in the training data.
- 'ts' (top speed) and 'rpr' (Racing Post Rating) may be hard to find in practice, it could require a subscription to the Racing Post.
- As a more general note, it might be useful to have a look at what data is available for the 2022 races, and engineer the features in the training data to match their distribution (e.g. make sure categories line up, etc.).

## Data Split

For development purposes, I'm splitting the data as follows:
- training set: all Galway races from 2005 to 2017.
- validation set: all Galway races from 2018.
- testing set: all Galway races from 2019.

When it comes time for deployment, I'll train and calibrate on all Galway data from 2005 to 2019. Hopefully, by then I'll have data for 2020 and 2021 as well. If I do, I'll test on these first, and then use them to train the final model.

## The Model

I want to use a boosted tree model because they usually achieve the best performance with tabular data. However, I also want the model to be well calibrated, as the prediction probabilities will be directly compared to implied probabilities from bookies to decide if I should bet and how big my stake should be.

Therefore, I'm going to use a two-stage model:
1. I train a lightGBM model, and the leaf indices of the trees will be used to essentially act as 'deep embeddings' of the training data.
2. These leaf indices are one-hot encoded and used to train a logistic regression model to output the probability of a horse winning.

Above is the high level view, but to detail each step in sequence:
1. Train a lightGBM model on training data:
    - I minimise focal loss here, as the minority class (winners) are of much more importance to me than the majority class (losers).
2. Predict the leaf indices of the training data with the lightGBM model.
3. One-hot encode the leaf indices.
4. Train a logistic regression model on the training data, using the one-hot encoded leaf indices as input.
    - I haven't tuned this layer's hyperparameters yet.
5. Use isotonic regression to calibrate the logistic regression model on the validation set.
6. Generate predictions for the train, validation and test sets to assess performance.

## Testing and Simulation

The model outputs a probability of a horse winning a race. In order to use this information, I go through the following steps:
1. Calculate the implied probability of a worse winning from the odds provided by the bookies.
2. Use the Kelly criterion to determine if I should bet, and if I should, how much I should stake.
3. Calculate a final stake using a fixed fraction (a half or a quarter) of the amount determined by the Kelly criterion.

In order to simulate how a series of races (or full race week) would have gone, I do the following:
1. Specify a starting bankroll (usually €100) and a fraction of Kelly bets to use (usually a half).
2. Calculate stakes and returns from the first race, and use that to update my bankroll.
3. Use the updated bankroll to calculate the stakes on the second race.
4. Rinse and repeat for all races.

At the moment, I don't have any restrictions on the size of the bets. Some things I could implement before this is actually put in production:
- Minimum bet size. Some suggested bets, particularly early in the week, are less than a euro. This isn't practically feasible, so it would be good to have a minimum value before a bet is placed (€1?, €2?, €5?).
- Bet rounding. Currently, I don't have any restrictions on the granularity of a bet. Obviously, I can't bet fractions of a cent in practice. But I also don't want to be specifying bets to the cent in the real world. At the very least, I'd want to be rounding to the nearest euro. I could even round to the nearest €5 or €10 as the bets got bigger. Something that adapts to the scale of the bet would be best (is it worth rounding to the nearest tenner if I'm betting hundreds of euro?). Also, rounding the nearest multiple may not necessarily be best - rounding down could be a more conservative approach.
- Maximum bet size. When I simulate an entire race week, I tend to see an exponential growth in bankroll. This means that the vast majority of money is made in the last few wins, because the stakes are so much higher. It would be much less taxing emotionally to have a maximum bet limit. However, it may limit my potential earnings significantly. It would be interesting to plot different winnings achieved with different maximum bet sizes. This limit may also be practically beneficial, as some bookies may stop accepting bets above a certain level. That's something I need to look into a bit more.