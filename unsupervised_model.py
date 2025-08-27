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

def find_chem_variance(df : pd.DataFrame, unit : str, cols : list[str]):

    

    pass


def hdbscan_clusters(df : pd.DataFrame, units : list[str], cols : list[str]):
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