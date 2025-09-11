import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import HDBSCAN, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from imblearn.over_sampling import ADASYN
import numpy as np
from scipy.stats import gaussian_kde

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

def kmeans_clusters(df : pd.DataFrame, units : list[str], cols : list[str], clusters : int = 2):
    df = df[df[Field.INTERPRETATION].isin(units)]
    df = df.dropna(subset=cols)

    scaler = StandardScaler()
    data = scaler.fit_transform(df[cols])

    kmeans = KMeans(
        n_clusters=2
    )

    df['predict'] = kmeans.fit_predict(data)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(df[['predict', Field.INTERPRETATION, Field.RELATEID]])
    print(df[['predict', Field.INTERPRETATION]].value_counts().reset_index())

    return df

def hdbscan_clusters(df : pd.DataFrame, units : list[str], cols : list[str], min_cluster_size : int = 4):
    df = df[df[Field.INTERPRETATION].isin(units)]

    scaler = StandardScaler()
    data = scaler.fit_transform(df[cols])

    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        allow_single_cluster=False,
    )

    df['predict'] = hdbscan.fit_predict(data)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(df[['predict', Field.INTERPRETATION, Field.RELATEID]])
    print(df[['predict', Field.INTERPRETATION]].value_counts().reset_index())

    return df

def display_clusters(df : pd.DataFrame, cols : list[str], units : list[str], show_axis : bool = True):
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
        print(predictions.shape)
        print(df.shape)

        np_array = df.loc[predictions == label].to_numpy()
        ax.scatter3D(xs=np_array[:, 0] * scalex, ys=np_array[:, 1] * scaley, zs=np_array[:, 2] * scalez, marker=marker_list[idx] if idx < len(marker_list) else 's'
                     , c=colors[predictions == label], cmap='viridis', norm=norm, s=30, label=str(label))

    if show_axis:
        for idx in range(n):
            ax.quiver(0, 0, 0, coef[idx, 0], coef[idx, 1], coef[idx, 2], color='r')
            ax.text(coef[idx, 0] * 1.15, coef[idx, 1] * 1.15, coef[idx, 2] * 1.15, cols[idx])

    ax.legend()

    sm = ScalarMappable(norm=norm, cmap='viridis_r')
    fig.colorbar(sm, ax=ax)
    plt.show()

def display_lith_clusters(df : pd.DataFrame, cols : list[str], units : list[str]):
    marker_list = ['o', 's', 'v', 'P', '*', 'D', 'h', '8']

    df = df[df[Field.INTERPRETATION].isin(units)]

    df = df.dropna()
    colors = df[Field.SAMPLE_DEPTH]
    norm = Normalize(vmin=df[Field.SAMPLE_DEPTH].min(), vmax=df[Field.SAMPLE_DEPTH].max())

    predictions = df['predict']
    df = df[cols]
    X = df[cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    score = X[:, 0:3]

    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]

    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scalez = 1.0 / (zs.max() - zs.min())

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    for idx, label in enumerate(predictions.unique()):
        print(predictions.shape)
        print(df.shape)

        np_array = df.loc[predictions == label].to_numpy()
        ax.scatter3D(xs=np_array[:, 0] * scalex, ys=np_array[:, 1] * scaley, zs=np_array[:, 2] * scalez, marker=marker_list[idx] if idx < len(marker_list) else 's'
                     , c=colors[predictions == label], cmap='viridis', norm=norm, s=30, label=str(label))

    ax.legend()

    sm = ScalarMappable(norm=norm, cmap='viridis_r')
    fig.colorbar(sm, ax=ax)
    plt.show()

def plot_pdf(df : pd.DataFrame, col : str, unit : str):
    df = df[df[Field.INTERPRETATION] == unit]
    X = df[col]

    masks = []
    colors = ['green', 'red', 'yellow', 'purple', 'black', 'orange', 'pink']

    for value in sorted(list(df['predict'].unique())):
        if value == -1:
            continue

        masks.append(df['predict'] == value)

    sns.kdeplot(X, fill=True, color='blue')

    for idx in range(len(masks)):
        sns.kdeplot(X[masks[idx]], fill=True, color=colors[idx], warn_singular=False)

    plt.show()

def find_chemical_stats(df : pd.DataFrame, cols : list[str]):
    for label in df['predict'].unique():
        mask = df['predict'] == label

        print(f'{label} | MEAN | RANGE')
        for chem in cols:
            chem_df = df.loc[mask, chem]

            print(f'{chem} | {round(chem_df.mean(), 4)} | {chem_df.min()}, {chem_df.max()}')

def plot_bar_charts(df : pd.DataFrame, col : str):

    std = df[col].std()
    mean = df[col].mean()

    bar_bottoms = [mean - std]
    bar_tops = [std * 2]
    bar_means = [mean]
    error_bottoms = [mean - df[col].min()]
    error_tops = [df[col].max() - mean]

    for value in sorted(list(df['predict'].unique())):
        if value == -1:
            continue

        std = df.loc[df['predict'] == value, col].std()
        mean = df.loc[df['predict'] == value, col].mean()

        bar_bottoms.append(mean - std)
        bar_tops.append(std * 2)
        bar_means.append(mean)
        error_bottoms.append(mean - df.loc[df['predict'] == value, col].min())
        error_tops.append(df.loc[df['predict'] == value, col].max() - mean)

    colors = ['blue', 'lime', 'red']
    x = ['Total', 'South', 'North']

    fig, ax = plt.subplots()
    bars = ax.bar(x, bar_tops, bottom=bar_bottoms, color=colors, edgecolor='black', alpha=.75)
    ax.errorbar(x, bar_means, yerr=[error_bottoms, error_tops], color='black', linestyle='none')

    for i, (bar, mean_val) in enumerate(zip(bars, bar_means)):
        x_center = bar.get_x() + bar.get_width() / 2
        ax.hlines(y=mean_val, xmin=bar.get_x(), xmax=bar.get_x() + bar.get_width(), colors='black', linestyles='--',
                  linewidth=1.5)

    ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()

def find_clusters(formation : str):
    df = data_refinement.load_and_combine()
    df['prev'] = df[Field.INTERPRETATION]
    df[Field.INTERPRETATION] = df[Field.INTERPRETATION].map(utils.FORMATION_MAP)

    cols = filter_chemicals(df, formation)

    if len(cols) < 3:
        print('Variance cutoff too high')
        print(cols)
        return

    df = hdbscan_clusters(df, [formation], cols)

    cond_df = df
    cond_df['predict'] = cond_df['predict'].replace({0: 0, 1: 0, 2: 1, 3: 1})

    sample_dict = {
        'OTS-4' : 0,
        'STL-1' : 0,
        'DOR-5' : 0,
        'DOR-3' : 0,
        'DOR-2' : 0,
        'DOR-1' : 0,
        'HBR-2' : 1,
        'SC9' : 1,
        'HBR-1' : 1,
        'OLR-1' : 1,
    }

    for sample in sample_dict.keys():
        cond_df.loc[cond_df[Field.SAMPLE_NUM].str.contains(sample), 'predict'] = sample_dict[sample]

    #cond_df['predict'] = cond_df['predict'].replace({0: 0, 1: 1, 2: 2, 3: 2, 4: 2})

    plot_pdf(cond_df, Field.CLEAR_PERCENTAGE, formation)

    #find_chemical_stats(cond_df, cols)
    #display_clusters(df, cols, [formation], True)

def find_lith_clusters(formation : str, units : list[str]):
    df = data_refinement.load_qdi_data()

    print(df[Field.UNIT].value_counts())
    return

    df = df[df[Field.UNIT].isin(units)]

    df = df[~((df[Field.CRYSTALLINE_PERCENTAGE] == 0) & (df[Field.CARBONATE_PERCENTAGE] == 0) & (df[Field.SHALE_PERCENTAGE] == 0))]
    df = df[~((df[Field.PRECAMBRIAN_PERCENTAGE] == 0) & (df[Field.PALEOZOIC_PERCENTAGE] == 0) & (df[Field.CRETACEOUS_PERCENTAGE] == 0))]
    df = df[~((df[Field.LIGHT_PERCENTAGE] == 0) & (df[Field.DARK_PERCENTAGE] == 0) & (df[Field.RED_PERCENTAGE] == 0))]
    df = df[~((df[Field.SAND_PERCENTAGE] == 0) & (df[Field.SILT_PERCENTAGE] == 0) & (df[Field.CLAY_PERCENTAGE] == 0))]

    df[Field.INTERPRETATION] = formation

    #df = hdbscan_clusters(df, [formation], [Field.CRYSTALLINE_PERCENTAGE, Field.CARBONATE_PERCENTAGE, Field.SHALE_PERCENTAGE,
    #                                        Field.PRECAMBRIAN_PERCENTAGE, Field.PALEOZOIC_PERCENTAGE, Field.CRETACEOUS_PERCENTAGE,
    #                                        Field.LIGHT_PERCENTAGE, Field.DARK_PERCENTAGE, Field.RED_PERCENTAGE], 4)

    df['predict'] = -1

    unique_ids = [
        "0000809697", "0000256714", "0000276730", "0000226764", "0000274221",
        "0000274222", "0000251485", "00Q0006453", "00Q0016320", "00Q0022789",
        "00Q0006488", "00Q0016318", "0000274220", "00Q0016419", "0000276731",
        "0000271891", "00Q0024230", "00Q0015388", "00Q0016317", "00Q0016433",
        "0000256717", "00Q0016372", "00Q0020574", "0000226763", "0000052010",
        "00Q0030242", "0000847291", "0000241869", "00Q0003032", "0000270359",
        '00Q0028555', '0000251922'
    ]

    df.loc[df[Field.RELATEID].isin(unique_ids), 'predict'] = 1
    df.loc[~df[Field.RELATEID].isin(unique_ids), 'predict'] = 0
    #df.loc[df[Field.SAMPLE_NUM].str.startswith(('HBR-2', 'SC9', 'HBR-1', 'OLR-1')), 'predict'] = 2

    plot_bar_charts(df, Field.PRECAMBRIAN_PERCENTAGE)

    #print(df.loc[df[Field.SAMPLE_NUM].str.startswith(('HBR-2', 'SC9', 'HBR-1', 'OLR-1')), Field.RELATEID].unique())
    #print(df.loc[df[Field.UNIT] == units[0], Field.RELATEID].unique())

    print(df.loc[df['predict'] == 1, Field.CRETACEOUS_PERCENTAGE].mean())
    print(df.loc[df['predict'] == 0, Field.CRETACEOUS_PERCENTAGE].mean())
    print(df[Field.CRETACEOUS_PERCENTAGE].mean())

    #plot_pdf(df, Field.CRETACEOUS_PERCENTAGE, formation)
    #display_lith_clusters(df, [Field.CRYSTALLINE_PERCENTAGE, Field.CARBONATE_PERCENTAGE, Field.SHALE_PERCENTAGE], [formation])

find_lith_clusters('Browerville', ['QBVU'])