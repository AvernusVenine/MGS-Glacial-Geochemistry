import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
from scipy import stats

import data_refinement
from utils import Field
import utils

def find_sse(y_true, y_pred):
    return sum(np.square(y_true - y_pred))

def find_ssr(y_true, y_pred):
    mean = y_true.mean()
    return sum(np.square(y_pred - mean))

def find_sst(y):
    mean = y.mean()
    return sum(np.square(y - mean))

def find_r_squared(df : pd.DataFrame, y_col : str, x_col : str = None):
    if x_col:
        X = df[[col for col in utils.CHEMICAL_COLS if col != x_col]]
    else:
        X = df[utils.CHEMICAL_COLS]
    y = df[y_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    reg = LinearRegression(
        fit_intercept=True
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)

    return find_ssr(y, y_pred), find_sse(y, y_pred), find_sst(y)

def find_p_values(df : pd.DataFrame, y_col : str):
    X = df[utils.CHEMICAL_COLS]
    y = df[y_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    reg = LinearRegression(
        fit_intercept=True
    )
    reg.fit(X, y)

    n = df.shape[0]
    beta_hat = [reg.intercept_] + reg.coef_.tolist()
    X1 = np.column_stack((np.ones(n), X))
    sigma_hat = np.sqrt(np.sum(np.square(y - X1 @ beta_hat)) / (n - X1.shape[1]))
    beta_cov = np.linalg.inv(X1.T @ X1)
    t_vals = beta_hat / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
    p_vals = stats.t.sf(np.abs(t_vals), n - X1.shape[1]) * 2
    p_vals = p_vals[1:]
    p_vals = p_vals.round(4)

    p_dict = {}

    for idx in range(len(p_vals)):
        p_dict[utils.CHEMICAL_COLS[idx]] = p_vals[idx]

    return p_dict

def find_all_p_values():
    df = data_refinement.load_and_combine()
    df = df.dropna(subset=utils.CHEMICAL_COLS + utils.LITHOLOGY_COLS)

    p_df = pd.DataFrame(columns=['Lithology'] + utils.CHEMICAL_COLS)

    for lith in utils.LITHOLOGY_COLS:
        p_dict = find_p_values(df, lith)
        p_dict['Lithology'] = lith

        p_df.loc[len(p_df)] = p_dict

    p_df.to_csv('data/p_values.csv')

def find_all_partial_r2():
    df = data_refinement.load_and_combine()
    df = df.dropna(subset=utils.CHEMICAL_COLS + utils.LITHOLOGY_COLS)

    score_df = pd.DataFrame(columns=['Lithology', 'Total'] + utils.CHEMICAL_COLS)

    for lith in utils.LITHOLOGY_COLS:
        score_dict = {}

        lith_ssr, _, lith_sst = find_r_squared(df, lith)
        total = round(lith_ssr / lith_sst, 4)

        score_dict['Lithology'] = lith
        score_dict['Total'] = total

        for chem in utils.CHEMICAL_COLS:
            ssr, sse, _ = find_r_squared(df, lith, chem)

            partial = round((lith_ssr - ssr) / sse, 4)
            score_dict[chem] = partial

        score_df.loc[len(score_df)] = score_dict

    score_df.to_csv('data/coef_of_det.csv')

find_all_p_values()