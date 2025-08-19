import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utils
from utils import Field

def show_correlation_matrix(df : pd.DataFrame):
    cm = df.corr()

    plt.imshow(cm, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()

    plt.xticks(range(len(cm)), cm.columns, rotation=45, ha='right')
    plt.yticks(range(len(cm)), cm.columns)
    plt.tight_layout()
    plt.show()

def show_3d_plot(df : pd.DataFrame, cols : list, unit : str):
    colors = df[Field.UNIT] == unit
    colors = colors.map({True: 'blue', False: 'red'})
    df = df[cols]

    np_array = df.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(xs=np_array[:, 0], ys=np_array[:, 1], zs=np_array[:, 2], c=colors)

    plt.show()

def show_pca_plot(df : pd.DataFrame, cols : list, unit : str):
    colors = df[Field.UNIT] == unit
    colors = colors.map({True : 'blue', False : 'red'})
    df = df[cols]

    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(df)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(xs=pca_array[:, 0], ys=pca_array[:, 1], zs=pca_array[:, 2], c=colors)

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
