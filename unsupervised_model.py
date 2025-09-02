import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from imblearn.over_sampling import ADASYN
import numpy as np
from scipy import stats

import data_refinement
import data_visualizer
import utils
from utils import Field

def filter_chemicals(df : pd.DataFrame, col : str):
    df = df[df[Field.INTERPRETATION] == col]
    df = df[utils.CHEMICAL_COLS]

    sklearn.set_config(transform_output='pandas')

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    selector = VarianceThreshold(
        threshold=.05
    )
    df = selector.fit_transform(df)

    return list(df.columns)

def hdbscan_clusters(df : pd.DataFrame, units : list[str], cols : list[str]):
    df['formation'] = df[Field.INTERPRETATION].map(utils.FORMATION_MAP)
    df = df[df['formation'].isin(units)]

    scaler = StandardScaler()
    data = scaler.fit_transform(df[cols])

    hdbscan = HDBSCAN(
        min_cluster_size=4,
        allow_single_cluster=False,
    )

    df['predict'] = hdbscan.fit_predict(data)

    pd.set_option('display.max_rows', None)

    print(df[['predict', Field.INTERPRETATION, Field.SAMPLE_NUM]])
    print(df[['predict', Field.INTERPRETATION]].value_counts().reset_index())

    return df

def display_clusters(df : pd.DataFrame, cols : list[str], units : list[str]):
    marker_list = ['o', 's', 'v', 'P', '*', 'D', 'h', '8']

    df = df[df[Field.INTERPRETATION].isin(units)]

    df = df.dropna()
    colors = df[Field.DEPTH]
    norm = Normalize(vmin=df[Field.DEPTH].min(), vmax=df[Field.DEPTH].max())

    predictions = df['predict']
    X = df[cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sklearn.set_config(transform_output='pandas')
    pca = PCA()
    df = pca.fit_transform(X)

    X = df.to_numpy()
    score = X[:, 0:3]

    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]

    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scalez = 1.0 / (zs.max() - zs.min())

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')

    coef = np.transpose(pca.components_[0:3, :])
    n = coef.shape[0]

    for idx, label in enumerate(predictions.unique()):

        np_array = df[predictions == label].to_numpy()
        ax.scatter3D(xs=np_array[:, 0] * scalex, ys=np_array[:, 1] * scaley, zs=np_array[:, 2] * scalez, marker=marker_list[idx], c=colors[predictions == label],
                     cmap='viridis', norm=norm, s=30, label=str(label))

    for idx in range(n):
        ax.quiver(0, 0, 0, coef[idx, 0], coef[idx, 1], coef[idx, 2], color='r')
        ax.text(coef[idx, 0] * 1.15, coef[idx, 1] * 1.15, coef[idx, 2] * 1.15, cols[idx])

    ax.legend()

    sm = ScalarMappable(norm=norm, cmap='viridis')
    fig.colorbar(sm, ax=ax)
    plt.show()

def find_clusters():
    df = data_refinement.load_geo_chem_data()

    cols = filter_chemicals(df, 'Browerville')

    df = hdbscan_clusters(df, ['Browerville'], cols)
    display_clusters(df, cols, ['Browerville'])

find_clusters()