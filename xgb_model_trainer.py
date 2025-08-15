import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot

import data_refinement
import utils
from utils import Field

# Trains a binary model based off of one or more units
def train_geo_chem_model_binary(df : pd.DataFrame, encoder : LabelEncoder, units : list):
    units = encoder.transform(units)

    y = df[Field.UNIT].tolist()
    y = [i in units for i in y]

    df = df.drop(columns=[Field.UNIT, Field.SAMPLE_NUM])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=127
    )

    model = xgboost.XGBClassifier(
        booster='dart',
        n_estimators=40,
        verbosity=0,

        rate_drop=.1,
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

def train_xgb_model(df : pd.DataFrame, encoder : LabelEncoder):
    print('TRAINING XGBCLASSIFIER MODEL')
    y = df[[Field.UNIT]].copy()

    df = df.drop(columns=[Field.UNIT, Field.RELATEID])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=127
    )

    # TODO: Refine this, currently drastically worsens the data
    #X_train, y_train = data_refinement.apply_smote(X_train, y_train)

    model = xgboost.XGBClassifier(
        booster='dart',
        n_estimators=100,
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

def show_feature_importance(model : XGBClassifier):
    plot_importance(model)
    pyplot.show()