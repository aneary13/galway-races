import optuna
import lightgbm as lgb
from optuna.integration import LightGBMTuner
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from loss import FocalLoss


# Create a class for the model
class HorseyGBM:

    # Initialise the model
    def __init__(self, target_label):
        self.num_features = []
        self.cat_features = []
        self.nan_features = []
        self.trainers = []
        self.jockeys = []
        self.processed_features = []
        self.imputer = IterativeImputer(sample_posterior=True,
                                        max_iter=50,
                                        initial_strategy='mean',
                                        verbose=0,
                                        random_state=0)
        self.target_label = target_label
        self.lgbm = None
        self.encoder = OneHotEncoder()
        self.lr = None
        self.calibrated_lr = None
        self.threshold = None

    # Method to perform all data pre-processing steps
    def preprocess(self, data, train=False):

        # Define feature lists
        if train:
            self.num_features = data.columns[((data.dtypes != 'object') & (data.columns != self.target_label))].tolist()
            self.cat_features = data.columns[data.dtypes == 'object'].tolist()
            self.nan_features = {x: f'{x}_nan' for x in data.loc[:, data.isna().any(axis=0)].columns}
            self.processed_features = self.num_features + self.cat_features + list(self.nan_features.values())
            all_trainers = data.loc[data[self.target_label] == 1, :].groupby('trainer').sum()
            self.trainers = list(all_trainers.loc[all_trainers['win'] >= 10, :].index)
            all_jockeys = data.loc[data[self.target_label] == 1, :].groupby('jockey').sum()
            self.jockeys = list(all_jockeys.loc[all_jockeys['win'] >= 10, :].index)

        # Initialise a copy of the data
        data = data.copy()

        # Create flags for features that have missing data
        for feat, nan_label in self.nan_features.items():
            data.loc[:, nan_label] = 0
            data.loc[data[feat].isna(), nan_label] = 1

        # Replace less common trainers and jockeys with 'Other'
        data.loc[:, 'trainer'] = data['trainer'].where(data['trainer'].isin(self.trainers), 'Other')
        data.loc[:, 'jockey'] = data['jockey'].where(data['jockey'].isin(self.jockeys), 'Other')

        # Fill missing categorical features with 'Unknown'
        data.loc[:, self.cat_features] = data.loc[:, self.cat_features].fillna('Unknown')
        data.loc[:, self.cat_features] = data.loc[:, self.cat_features].astype('category')

        # Fit an imputer on training set and transform data if training
        if train:
            data.loc[:, self.num_features] = self.imputer.fit_transform(data.loc[:, self.num_features])

        # Use pre-fitted imputer to transform data if not training
        else:
            data.loc[:, self.num_features] = self.imputer.transform(data.loc[:, self.num_features])

        # Return pre-processed data
        return data.loc[:, self.processed_features + [self.target_label]]

    # Method to train a model on the data
    def fit(self, train, valid):

        # Pre-process training and validation data
        train = self.preprocess(train, train=True)
        valid = self.preprocess(valid.copy(deep=True), train=False)

        # Create training and validation data
        train_dataset = lgb.Dataset(train.drop(self.target_label, axis=1),
                                    label=train[self.target_label])
        valid_dataset = lgb.Dataset(valid.drop(self.target_label, axis=1),
                                    label=valid[self.target_label])

        # Initialise parameters for model
        fl = FocalLoss()
        eval_result = {}
        params = {
            'learning_rate': 0.05,
            'metric': 'focal_loss',
            'force_col_wise': 'True'
        }

        # Train and tune a lightGBM model
        study = optuna.create_study(direction='minimize')
        tuner = LightGBMTuner(params=params,
                              train_set=train_dataset,
                              valid_sets=(train_dataset, valid_dataset),
                              valid_names=('train', 'val'),
                              num_boost_round=10000,
                              early_stopping_rounds=100,
                              verbose_eval=100,
                              study=study,
                              fobj=fl.f_obj,
                              feval=fl.f_eval,
                              keep_training_booster=True,
                              callbacks=[lgb.callback.record_evaluation(eval_result)])
        tuner.run()
        self.lgbm = tuner.get_best_booster()

        # Train a logistic regression model on the leaf indices output by the lightGBM model
        train_leaves = self.lgbm.predict(train.drop(self.target_label, axis=1), pred_leaf=True)
        train_leaves_encoded = self.encoder.fit_transform(train_leaves)
        self.lr = LogisticRegression(solver='sag', C=10e-3, fit_intercept=False)
        self.lr.fit(train_leaves_encoded, train[self.target_label])

        # Generate leaf indices for validation set
        val_leaves = self.lgbm.predict(valid.drop(self.target_label, axis=1), pred_leaf=True)
        val_leaves_encoded = self.encoder.transform(val_leaves)

        # Calibrate the logistic regression model
        self.calibrated_lr = CalibratedClassifierCV(base_estimator=self.lr, method='isotonic', cv='prefit')
        self.calibrated_lr.fit(val_leaves_encoded, valid[self.target_label])

    # Method to make predictions on a dataset
    def predict(self, data):

        # Pre-process the data
        data = self.preprocess(data, train=False)

        # Drop target label
        data = data.drop(self.target_label, axis=1)

        # Make predictions on the dataset
        leaves = self.lgbm.predict(data, pred_leaf=True)
        encoded_leaves = self.encoder.transform(leaves)
        pred = self.calibrated_lr.predict_proba(encoded_leaves)

        # Return prediction probabilities for the positive class
        return pred[:, 1]
