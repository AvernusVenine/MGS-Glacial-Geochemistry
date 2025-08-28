import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
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

def train_minimal_models():
    df = data_refinement.load_and_combine()
    df = df.dropna(subset=utils.CHEMICAL_COLS + utils.LITHOLOGY_COLS)
    coef_df, p_df = load_data(True)

    sig_chems = find_significant_chemicals(p_df, .01)

    for lith in utils.LITHOLOGY_COLS:
        X = df[sig_chems[lith]]
        y = df[lith]

        if not sig_chems[lith]:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        reg = LinearRegression(
            fit_intercept=True
        )
        reg.fit(X, y)

        print(f'{lith} | {reg.score(X, y)} | {sig_chems[lith]}')

def find_coefficients(df : pd.DataFrame, y_col : str, positive : bool = False):
    X = df[utils.CHEMICAL_COLS]
    y = df[y_col]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    reg = LinearRegression(
        fit_intercept=True,
        positive=positive
    )
    reg.fit(X, y)

    coef_dict = {}

    for idx in range(len(utils.CHEMICAL_COLS)):
        coef_dict[utils.CHEMICAL_COLS[idx]] = round(reg.coef_[idx], 4)

    return coef_dict

def find_r_squared(X, y, x_col : str = None):
    if x_col:
        X = X[[col for col in list(X.columns) if col != x_col]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    reg = LinearRegression(
        fit_intercept=True,
        positive=True
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
        fit_intercept=True,
        positive=True
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

    p_df.to_csv('data/p_values_pos.csv')

def find_all_coefficients():
    df = data_refinement.load_and_combine()
    df = df.dropna(subset=utils.CHEMICAL_COLS + utils.LITHOLOGY_COLS)

    ceof_df = pd.DataFrame(columns=['Lithology'] + utils.CHEMICAL_COLS)

    for lith in utils.LITHOLOGY_COLS:
        coef_dict = find_coefficients(df, lith, True)
        coef_dict['Lithology'] = lith
        ceof_df.loc[len(ceof_df)] = coef_dict

    ceof_df.to_csv('data/coefficients_pos.csv')

def find_all_partial_r2():
    df = data_refinement.load_chemical_ratios()
    df = df.dropna()

    df = df.drop(columns=[Field.SAMPLE_NUM, Field.INTERPRETATION, Field.DEPTH])
    partial_df = df.drop(columns=utils.LITHOLOGY_COLS + utils.CHEMICAL_COLS)

    score_df = pd.DataFrame(columns=['Lithology', 'Total'] + list(partial_df.columns))

    for lith in utils.LITHOLOGY_COLS:
        lith = Field.CARBONATE_PERCENTAGE
        print(lith)

        score_dict = {}

        lith_ssr, _, lith_sst = find_r_squared(partial_df, df[lith])
        total = round(lith_ssr / lith_sst, 10)

        score_dict['Lithology'] = lith
        score_dict['Total'] = total

        for chem in list(partial_df.columns):
            print(chem)
            ssr, sse, _ = find_r_squared(partial_df, df[lith], chem)

            partial = round((lith_ssr - ssr) / sse, 10)
            score_dict[chem] = partial

        score_df.loc[len(score_df)] = score_dict
        break

    score_df.to_csv('data/ratio_coef_of_det.csv')

def find_significant_chemicals(df : pd.DataFrame, cutoff : float = .05):
    sig_dict = {}

    for _, row in df.iterrows():
        sig_list = []

        for chem in utils.CHEMICAL_COLS:

            if row[chem] == 1:
                continue

            if row[chem] < cutoff or (1 - cutoff) < row[chem]:
                sig_list.append(chem)

        sig_dict[row['Lithology']] = sig_list

    return sig_dict

def chemical_pca_analysis():
    df = data_refinement.load_geo_chem_data()
    df = df[utils.CHEMICAL_COLS]

    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(df)

    pca_df = pd.DataFrame(columns=['PCA'] + utils.CHEMICAL_COLS)


    for comp in range(20):
        pca_dict = {}
        for idx in range(len(pca.components_[comp])):
            pca_dict['PCA'] = comp + 1
            pca_dict[utils.CHEMICAL_COLS[idx]] = pca.components_[comp][idx]

        sorted_dict_desc = dict(sorted(pca_dict.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict_desc)
        print(pca.explained_variance_ratio_[comp])
        pca_df.loc[len(pca_df)] = pca_dict

    cov_df = pd.DataFrame(pca.get_covariance(), columns=utils.CHEMICAL_COLS, index=utils.CHEMICAL_COLS)
    cov_df = cov_df.round(4)

    cov_df.to_csv('data/chemical_covariance.csv')
    pca_df.to_csv('data/chemical_pca.csv', index=False)

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

    print(pca.get_covariance())

    pca_df.to_csv('data/ratio_pca.csv', index=False)

def load_data(positive : bool = False):
    if positive:
        coef_df = pd.read_csv('data/linear_regression/coef_of_det_pos.csv')
        p_df = pd.read_csv('data/linear_regression/p_values_pos.csv')

        return coef_df, p_df
    else:
        coef_df = pd.read_csv('data/linear_regression/coef_of_det.csv')
        p_df = pd.read_csv('data/linear_regression/p_values.csv')

        return coef_df, p_df
