import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from imblearn.over_sampling import ADASYN
import numpy as np
from scipy import stats

import utils
from utils import Field

def hdbscan_clusters(df : pd.DataFrame, units : list[str]):
    df['formation'] = df[Field.INTERPRETATION].map(utils.FORMATION_MAP)
    df = df[df['formation'].isin(units)]

    scaler = RobustScaler()
    data = scaler.fit_transform(df[utils.CHEMICAL_COLS + [Field.DEPTH]])

    hdbscan = HDBSCAN(
        min_cluster_size=4,
        allow_single_cluster=True,
    )

    df['predict'] = hdbscan.fit_predict(data)

    pd.set_option('display.max_rows', None)

    print(df[['predict', Field.INTERPRETATION, Field.SAMPLE_NUM]])
    print(df[['predict', Field.INTERPRETATION]].value_counts().reset_index())

def apply_smote(X : pd.DataFrame, y : pd.DataFrame):
    adasyn = ADASYN(
        sampling_strategy='not majority',
        random_state=127,
        n_neighbors=5,
    )

    X, y = adasyn.fit_resample(X, y)
    return X, y

def load_qdi_data():
    df = pd.read_csv(utils.QDI_SAMPLE_PATH)
    return df

def load_geo_chem_data(refine = True):
    df = pd.read_csv(utils.GEO_CHEM_PATH)
    return remove_outliers(df) if refine else df

def remove_outliers(df : pd.DataFrame):
    return df[np.abs(stats.zscore(df[utils.CHEMICAL_COLS]) < 5).all(axis=1)]

def scale_df(df : pd.DataFrame, cols : list):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df

def label_encode_units(df : pd.DataFrame, field : str):
    encoder = LabelEncoder()
    encoder.fit_transform(df[field])

    df[field] = encoder.transform(df[field])

    return df, encoder
