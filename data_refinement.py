import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN

import utils
from utils import Field

def perform_pca(df : pd.DataFrame):
    df = df[utils.CHEMICAL_COLS]

    pca = PCA(n_components=3)
    pca.fit

    pass

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

def load_geo_chem_data():
    df = pd.read_csv(utils.GEO_CHEM_PATH)
    return df

def scale_df(df : pd.DataFrame, cols : list):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df

def label_encode_units(df : pd.DataFrame):
    encoder = LabelEncoder()
    encoder.fit_transform(df[Field.UNIT])

    df[Field.UNIT] = encoder.transform(df[Field.UNIT])

    return df, encoder
