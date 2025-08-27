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

def show_correlation_matrix(df : pd.DataFrame, y_cols : list = None):
    df = df.drop(columns=[Field.INTERPRETATION, Field.SAMPLE_NUM, Field.DEPTH])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    cm = df.corr()
    if y_cols:
        cm = cm[y_cols]
        cm = cm.transpose()

    print(cm.shape)

    plt.imshow(cm, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()

    plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha='right')

    if y_cols:
        plt.yticks(range(len(y_cols)), y_cols)
    else:
        plt.yticks(range(len(cm)), cm.columns)

    plt.tight_layout()
    plt.show()

def show_2d_plot(df : pd.DataFrame, cols : list):
    df = df[cols]

    np_array = df.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])

    ax.scatter(x=np_array[:, 0], y=np_array[:, 1])

    plt.show()

def show_3d_plot(df : pd.DataFrame, cols : list, unit : str = None):
    colors = df[Field.INTERPRETATION] == unit
    if unit:
        colors = colors.map({True: 'blue', False: 'red'})
    df = df[cols]

    np_array = df.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    if unit:
        ax.scatter3D(xs=np_array[:, 0], ys=np_array[:, 1], zs=np_array[:, 2], c=colors)
    else:
        ax.scatter3D(xs=np_array[:, 0], ys=np_array[:, 1], zs=np_array[:, 2])

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

def show_sample_pca_depth_plot(df : pd.DataFrame, cols : list, samples : list[str], unit : str = None, samples_only : bool = False):
    marker_list = ['o', 'v', 'P', '*', 'D', 'h']

    df = df.dropna()
    if unit:
        df = df[df[Field.INTERPRETATION] == unit]

    colors = df[Field.DEPTH]
    norm = Normalize(vmin=df[Field.DEPTH].min(), vmax=df[Field.DEPTH].max())
    interp = df[Field.SAMPLE_NUM]

    df = df[cols]

    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(df[cols])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    idx = 0
    for sample in samples:
        mask = interp.str.startswith(sample)
        df.loc[mask, Field.SAMPLE_NUM] = sample

        ax.scatter3D(xs=pca_array[mask, 0], ys=pca_array[mask, 1], zs=pca_array[mask, 2], marker=marker_list[idx], c=colors[mask],
                     cmap='viridis', norm=norm, s=60)

        idx += 1

    if not samples_only:
        mask = ~interp.str.startswith(tuple(samples))

        ax.scatter3D(xs=pca_array[mask, 0], ys=pca_array[mask, 1], zs=pca_array[mask, 2], marker='s',
                     c=colors[mask],
                     cmap='viridis', norm=norm, s=15)

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
