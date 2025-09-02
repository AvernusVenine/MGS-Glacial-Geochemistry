import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import data_refinement
import linear_regression
from utils import Field
import utils

def fit_linear_regression(x_cols : list[str], y_col : str):
    df = data_refinement.load_chemical_ratios()
    df = df.drop(
        columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH])
    df = df.dropna()

    X = df[x_cols]
    y = df[y_col]

    reg = LinearRegression(
        positive=True
    )
    reg.fit(X, y)

    coef_dict = {}

    for idx in range(len(x_cols)):
        if reg.coef_[idx] == 0:
            continue

        coef_dict[x_cols[idx]] = round(reg.coef_[idx], 4)

    series = pd.Series(coef_dict).abs().sort_values(ascending=False)
    print(reg.score(X, y))
    print(series[0:20])

    return coef_dict

def ratio_lasso_regression(y_col : str, alpha : float = 1):
    df = data_refinement.load_chemical_ratios()
    df = df.drop(
        columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH])
    df = df.dropna()

    y = df[y_col]
    X = df.drop(columns=utils.LITHOLOGY_COLS)

    lasso = Lasso(
        positive=True,
        alpha=alpha
    )

    lasso.fit(X, y)

    coef_dict = {}

    for idx in range(len(lasso.coef_)):
        if lasso.coef_[idx] == 0:
            continue

        coef_dict[X.columns[idx]] = lasso.coef_[idx]

    series = pd.Series(coef_dict).abs().sort_values(ascending=False)
    print(lasso.score(X, y))
    print(series[0:20])

    return list(coef_dict.keys())

def ratio_pls_regression():
    df = data_refinement.load_chemical_ratios()
    df = df.drop(
        columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH])
    df = df.dropna()

    y = df[utils.MIN_LITH_COLS].copy()
    X = df.drop(columns=utils.LITHOLOGY_COLS)

    pls = PLSRegression(
        n_components=20
    )
    pls.fit(X, y)

    print(pls.x_loadings_)

    for lith in range(len(pls.coef_)):
        coef_dict = {}
        #coef_dict['Lithology'] = utils.LITHOLOGY_COLS[lith]
        #print(utils.LITHOLOGY_COLS[lith])

        for chem in range(len(pls.coef_[lith])):
            coef_dict[df.columns[chem]] = pls.coef_[lith][chem]

        series = pd.Series(coef_dict).abs()
        #print(series)

def ratio_vif():
    df = data_refinement.load_chemical_ratios()
    df = df.drop(
        columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH] + utils.LITHOLOGY_COLS + utils.CHEMICAL_COLS)
    df = df.dropna()

    vif_df = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

    print(vif_df)

def find_high_ratio_correlation(numer : str, denom : str = None, elements_only : bool = False):
    df = data_refinement.load_chemical_ratios()
    df = df.drop(
        columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH])
    df = df.dropna()

    col = numer
    if denom:
        col = f'{numer} / {denom}'

    if elements_only:
        df = df[utils.CHEMICAL_COLS]

    cm = df.corr().abs()
    cm = cm.loc[col].sort_values(ascending=False)

    if denom:
        cm = cm[~cm.index.str.contains(col)]
        cm = cm[~cm.index.str.contains(denom)]

    print(cm[0:20])

def ratio_pca_analysis():
    df = data_refinement.load_chemical_ratios()
    df = df.drop(columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH] + utils.LITHOLOGY_COLS + utils.CHEMICAL_COLS)
    df = df.dropna()

    pca = PCA()
    pca.fit(df)

    pca_df = pd.DataFrame(columns=['PCA'] + list(df.columns))

    for comp in range(20):
        pca_dict = {}
        for idx in range(len(pca.components_[comp])):
            pca_dict['PCA'] = comp + 1
            pca_dict[list(df.columns)[idx]] = pca.components_[comp][idx]

        sorted_dict_desc = dict(sorted(pca_dict.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict_desc)
        print(pca.explained_variance_ratio_[comp])
        pca_df.loc[len(pca_df)] = pca_dict

    pca_df.to_csv('data/ratio_pca.csv', index=False)

def ratio_correlation_analysis():
    df = data_refinement.load_chemical_ratios()
    df = df.drop(
        columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH] + utils.LITHOLOGY_COLS)
    df = df.dropna()

    chem_cm = df[utils.CHEMICAL_COLS].corr()
    chem_cm.to_csv('chemical_correlations.csv')

    ratio_cm = df.drop(columns=utils.CHEMICAL_COLS)
    ratio_cm.to_csv('ratio_correlations.csv')

def find_ratio_partial_r2(lith : str):
    df = data_refinement.load_chemical_ratios()
    df = df.dropna()

    cols = ratio_lasso_regression(lith ,.25)
    cols = fit_linear_regression(cols, lith)

    lith_ssr, _, lith_sst = linear_regression.find_r_squared(df[cols.keys()], df[lith])
    total = round(lith_ssr / lith_sst, 10)

    score_dict = {}
    #score_dict['Lithology'] = Field.CARBONATE_PERCENTAGE
    #score_dict['Total'] = total
    print(total)

    for chem in list(cols.keys()):
        ssr, sse, _ = linear_regression.find_r_squared(df[cols.keys()], df[lith], chem)

        partial = round((lith_ssr - ssr) / sse, 10)
        score_dict[chem] = partial

    series = pd.Series(score_dict).sort_values(ascending=False)
    print(series)

find_ratio_partial_r2(Field.PRECAMBRIAN_PERCENTAGE)
#find_high_ratio_correlation(Field.CS_PPM, elements_only=True)
#find_high_ratio_correlation(Field.CS_PPM, Field.CU_PPM)