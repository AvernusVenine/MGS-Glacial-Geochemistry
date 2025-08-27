import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from imblearn.over_sampling import ADASYN
import numpy as np
from scipy import stats

import data_visualizer
import utils
from utils import Field

def compare_lith_chem():
    df = load_and_combine()

    data_visualizer.show_correlation_matrix(df, [Field.SAND_PERCENTAGE, Field.SILT_PERCENTAGE, Field.CLAY_PERCENTAGE,
                                                 Field.CRYSTALLINE_PERCENTAGE, Field.CARBONATE_PERCENTAGE, Field.SHALE_PERCENTAGE,
                                                 Field.PRECAMBRIAN_PERCENTAGE, Field.PALEOZOIC_PERCENTAGE, Field.CRETACEOUS_PERCENTAGE,
                                                 Field.LIGHT_PERCENTAGE, Field.DARK_PERCENTAGE, Field.RED_PERCENTAGE])
    #data_visualizer.show_2d_plot(df, [Field.SAND_PERCENTAGE, Field.NA_PERCENTAGE])

def load_and_combine():
    lith_df = load_qdi_data()
    df = load_geo_chem_data()

    lith_df = lith_df.drop_duplicates(subset=[Field.SAMPLE_NUM])
    df = df.drop_duplicates(subset=[Field.SAMPLE_NUM])

    df[Field.SAND_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.SAND_PERCENTAGE])
    df[Field.SILT_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.SILT_PERCENTAGE])
    df[Field.CLAY_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.CLAY_PERCENTAGE])

    df[Field.CRYSTALLINE_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.CRYSTALLINE_PERCENTAGE])
    df[Field.CARBONATE_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.CARBONATE_PERCENTAGE])
    df[Field.SHALE_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.SHALE_PERCENTAGE])

    df[Field.PRECAMBRIAN_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.PRECAMBRIAN_PERCENTAGE])
    df[Field.PALEOZOIC_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.PALEOZOIC_PERCENTAGE])
    df[Field.CRETACEOUS_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.CRETACEOUS_PERCENTAGE])

    df[Field.LIGHT_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.LIGHT_PERCENTAGE])
    df[Field.DARK_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.DARK_PERCENTAGE])
    df[Field.RED_PERCENTAGE] = df[Field.SAMPLE_NUM].map(lith_df.set_index(Field.SAMPLE_NUM)[Field.RED_PERCENTAGE])

    return df

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
