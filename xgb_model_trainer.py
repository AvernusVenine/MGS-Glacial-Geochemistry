import xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
import numpy as np
import ray.tune as tune

import data_refinement
import utils
from utils import Field

def find_key_chemicals(units : list, trials : int, against : list = None):
    df = data_refinement.load_geo_chem_data()

    if against:
        df = df[df[Field.INTERPRETATION].isin(against)]

    df, encoder = data_refinement.label_encode_units(df, Field.INTERPRETATION)

    chem_dict = {}

    for _ in range(trials):
        model = train_geo_chem_model_binary(df, encoder, units)

        imp_df = pd.DataFrame(model.get_booster().get_score(importance_type='gain'), index=[0])
        imp_df = imp_df.transpose()

        imp_df = imp_df.sort_values(by=0).head(3)

        for idx, row in imp_df.iterrows():

            if str(idx) not in chem_dict.keys():
                chem_dict[str(idx)] = 1
            else:
                chem_dict[str(idx)] += 1

    sorted_dict = {}
    for key in sorted(chem_dict, key=chem_dict.get):
        sorted_dict[key] = chem_dict[key]

    return sorted_dict

# Trains a binary model based off of one or more units
def train_geo_chem_model_binary(df : pd.DataFrame, encoder : LabelEncoder, units : list):
    units = encoder.transform(units)

    y = df[Field.INTERPRETATION].tolist()
    y = [i in units for i in y]

    df = df.drop(columns=[Field.INTERPRETATION, Field.SAMPLE_NUM, Field.DEPTH])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        #random_state=314
    )

    model = xgboost.XGBClassifier(
        #booster='dart',
        n_estimators=500,
        verbosity=0,
        early_stopping_rounds=20,

        #rate_drop=.1,
        #normalize_type='forest',
        #sample_type='weighted',

        #tree_method='hist',
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test)

    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return model

# Trains a general model based off of geochemistry, however it is very easy to overfit due to small dataset size
def train_geo_chem_model(df : pd.DataFrame, encoder : LabelEncoder):
    y = df[[Field.UNIT]].copy()

    df = df.drop(columns=[Field.UNIT, Field.SAMPLE_NUM])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=2
    )

    model = xgboost.XGBClassifier(
        booster='dart',
        n_estimators=200,
        verbosity=2,

        rate_drop = .1,
        normalize_type='forest',
        sample_type='weighted',

        tree_method='hist',
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return model

def hyperparameter_train_classifier(config):
    df = data_refinement.load_qdi_data()
    encoder = LabelEncoder()

    y = encoder.fit_transform(df[Field.UNIT])

    df = df.drop(columns=[Field.UNIT, Field.RELATEID, Field.SAMPLE_NUM])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=127
    )

    model = xgboost.XGBClassifier(
        booster='dart',
        verbosity=2,

        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        normalize_type=config['normalize_type'],
        sample_type=config['sample_type'],
        learning_rate=config['learning_rate'],
        subsample=config['subsample'],
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)

    tune.report({
        'accuracy': accuracy_score(y_test, y_pred)
    })

def train_xgb_model(df : pd.DataFrame):
    print('TRAINING XGBCLASSIFIER MODEL')
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[Field.UNIT])

    df = df.drop(columns=[Field.UNIT, Field.RELATEID, Field.SAMPLE_NUM])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=127
    )

    # TODO: Refine this, currently drastically worsens the data
    #X_train, y_train = data_refinement.apply_smote(X_train, y_train)

    model = xgboost.XGBClassifier(
        booster='dart',
        n_estimators=200,
        verbosity=2,
        max_depth=6,

        rate_drop = .10,
        normalize_type='forest',
        sample_type='weighted',

        tree_method='hist',
        random_state=127,
    )

    cv_scores = cross_val_score(model, df, y)

    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Average Score: {np.mean(cv_scores)}')

    return model

def show_feature_importance(model : XGBClassifier, importance_type : str):
    plot_importance(model, importance_type=importance_type)
    pyplot.show()

def get_unused_features(model : XGBClassifier):
    importance_score = model.get_booster().get_score(importance_type='weight')
    unused_features = list(set(utils.CHEMICAL_COLS) - set(importance_score.keys()))

    return unused_features