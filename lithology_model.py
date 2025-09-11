import xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
import numpy as np
import ray.tune as tune

import data_refinement
import utils
from utils import Field

def train_decision_tree():
    df = data_refinement.load_qdi_data()
    df = df.dropna(subset=[Field.UNIT])

    df = df[~((df[Field.CRYSTALLINE_PERCENTAGE] == 0) & (df[Field.CARBONATE_PERCENTAGE] == 0) & (df[Field.SHALE_PERCENTAGE] == 0))]
    df = df[~((df[Field.PRECAMBRIAN_PERCENTAGE] == 0) & (df[Field.PALEOZOIC_PERCENTAGE] == 0) & (df[Field.CRETACEOUS_PERCENTAGE] == 0))]
    df = df[~((df[Field.LIGHT_PERCENTAGE] == 0) & (df[Field.DARK_PERCENTAGE] == 0) & (df[Field.RED_PERCENTAGE] == 0))]
    df = df[~((df[Field.SAND_PERCENTAGE] == 0) & (df[Field.SILT_PERCENTAGE] == 0) & (df[Field.CLAY_PERCENTAGE] == 0))]

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
        n_estimators=150,
        max_depth=10,
        rate_drop=.10,
        sample_type='weighted',
        normalize_type='tree'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

def train_neural_network():
    df = data_refinement.load_qdi_data()
    df = df.dropna(subset=[Field.UNIT])

    df = df[~((df[Field.CRYSTALLINE_PERCENTAGE] == 0) & (df[Field.CARBONATE_PERCENTAGE] == 0) & (df[Field.SHALE_PERCENTAGE] == 0))]
    df = df[~((df[Field.PRECAMBRIAN_PERCENTAGE] == 0) & (df[Field.PALEOZOIC_PERCENTAGE] == 0) & (df[Field.CRETACEOUS_PERCENTAGE] == 0))]
    df = df[~((df[Field.LIGHT_PERCENTAGE] == 0) & (df[Field.DARK_PERCENTAGE] == 0) & (df[Field.RED_PERCENTAGE] == 0))]
    df = df[~((df[Field.SAND_PERCENTAGE] == 0) & (df[Field.SILT_PERCENTAGE] == 0) & (df[Field.CLAY_PERCENTAGE] == 0))]

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[Field.UNIT])

    df = df.drop(columns=[Field.UNIT, Field.RELATEID, Field.SAMPLE_NUM])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=0.2,
        random_state=127
    )

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',

        verbose=True,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

train_neural_network()