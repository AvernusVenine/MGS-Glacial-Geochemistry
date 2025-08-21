import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats

import utils
from utils import Field

def show_1d_plot(df : pd.DataFrame, col : str, units : list[str], units_only : bool = False):

    if units_only:
        df = df[df[Field.INTERPRETATION].isin(units)]

    pass

def show_lda_plot(df : pd.DataFrame, cols : list, units : list[str], units_only : bool = False):
    color_list = ['blue', 'green', 'black', 'purple', 'pink', 'orange', 'gray']
    color_map = {}

    df = df.dropna()

    if units_only:
        df = df[df[Field.INTERPRETATION].isin(units)]
        color_list.append('red')

    y = df[Field.INTERPRETATION]
    X = df[cols]

    lda = LinearDiscriminantAnalysis(
        n_components=3
    )
    lda_array = lda.fit_transform(X, y)

    idx = 0
    for unit in units:
        color_map[unit] = color_list[idx]
        idx += 1

    colors = y.map(color_map)
    colors = colors.fillna('red')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(xs=lda_array[:, 0], ys=lda_array[:, 1], zs=lda_array[:, 2], c=colors, s=20)

    plt.show()


def show_correlation_matrix(df : pd.DataFrame):
    df = df.drop(columns=[Field.INTERPRETATION, Field.SAMPLE_NUM, Field.DEPTH])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    cm = df.corr()

    plt.imshow(cm, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()

    plt.xticks(range(len(cm)), cm.columns, rotation=45, ha='right')
    plt.yticks(range(len(cm)), cm.columns)
    plt.tight_layout()
    plt.show()

def show_3d_plot(df : pd.DataFrame, cols : list, unit : str):
    colors = df[Field.INTERPRETATION] == unit
    colors = colors.map({True: 'blue', False: 'red'})
    df = df[cols]

    np_array = df.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    ax.scatter3D(xs=np_array[:, 0], ys=np_array[:, 1], zs=np_array[:, 2], c=colors)

    plt.show()

def show_3d_depth_plot(df : pd.DataFrame, cols : list, units : list[str], units_only : bool = False):
    marker_list = ['o', 'v', 'P', '*', 'D', 'h', '8']

    df = df.dropna()

    if units_only:
        df = df[df[Field.INTERPRETATION].isin(units)]
        marker_list.append('s')

    colors = df[Field.DEPTH]
    norm = Normalize(vmin=df[Field.DEPTH].min(), vmax=df[Field.DEPTH].max())
    interp = df[Field.INTERPRETATION]

    df = df[cols]

    np_array = df.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    idx = 0
    for unit in units:
        mask = interp == unit

        ax.scatter3D(xs=np_array[mask, 0], ys=np_array[mask, 1], zs=np_array[mask, 2], marker=marker_list[idx], c=colors[mask],
                     cmap='viridis', norm=norm, s=20)

        idx += 1

    sm = ScalarMappable(norm=norm, cmap='viridis')
    fig.colorbar(sm, ax=ax)
    plt.show()

def show_pca_depth_plot(df : pd.DataFrame, cols : list, units : list[str], units_only : bool = False):
    marker_list = ['o', 'v', 'P', '*', 'D', 'h', '8']

    # Necessary, as some of the samples have no depth recorded
    df = df.dropna()

    if units_only:
        df = df[df[Field.INTERPRETATION].isin(units)]
        marker_list.append('s')

    colors = df[Field.DEPTH]
    norm = Normalize(vmin=df[Field.DEPTH].min(), vmax=df[Field.DEPTH].max())
    interp = df[Field.INTERPRETATION]

    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(df[cols])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    idx = 0
    for unit in units:
        mask = interp == unit

        ax.scatter3D(xs=pca_array[mask, 0], ys=pca_array[mask, 1], zs=pca_array[mask, 2], marker=marker_list[idx], c=colors[mask],
                     cmap='viridis', norm=norm, s=35)

        idx += 1

    if ~units_only:
        mask = ~interp.isin(units)

        ax.scatter3D(xs=pca_array[mask, 0], ys=pca_array[mask, 1], zs=pca_array[mask, 2], marker='v',
                     c=colors[mask],
                     cmap='viridis', norm=norm, s=15)

    sm = ScalarMappable(norm=norm, cmap='viridis')

    fig.colorbar(sm, ax=ax)
    plt.show()


def show_pca_plot(df : pd.DataFrame, cols : list, units : list[str], units_only : bool = False):
    color_list = ['blue', 'green', 'black', 'purple', 'pink']
    color_map = {}

    if units_only:
        df = df[df[Field.INTERPRETATION].isin(units)]
        color_list.append('red')

    colors = df[Field.INTERPRETATION].copy()

    idx = 0
    for unit in units:
        color_map[unit] = color_list[idx]
        idx += 1

    colors = colors.map(color_map)
    colors = colors.fillna('red')

    df = df[cols]

    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(df)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(xs=pca_array[:, 0], ys=pca_array[:, 1], zs=pca_array[:, 2], c=colors, s=20)

    plt.show()

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
