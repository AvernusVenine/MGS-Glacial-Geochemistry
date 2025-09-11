import pandas
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
import data_refinement
import data_visualizer
import utils
import xgb_model_trainer
from data_refinement import Field
import ray.tune as tune
from ray.tune.search.optuna import OptunaSearch

import linear_regression


def train_geo_chem():
    df = data_refinement.load_geo_chem_data()

    df = df[df[Field.INTERPRETATION].isin(['GT1', 'GT2', 'GT3', 'GT4', 'GT5'])]

    df, encoder = data_refinement.label_encode_units(df, Field.INTERPRETATION)

    model = xgb_model_trainer.train_geo_chem_model_binary(df, encoder, ['GT4'])
    #model = xgb_model_trainer.train_geo_chem_model(df, encoder)
    #data_refinement.show_pca_plot(old_df, xgb_model_trainer.get_unused_features(model), 'QBVU')

    plot_importance(model, importance_type='gain')
    plot_importance(model, importance_type='weight')
    plt.show()

def find_key_chemicals():
    key_chem = xgb_model_trainer.find_key_chemicals(['Ivanhoe'], 200)

    key_chem = {k: v for k, v in key_chem.items() if v >= 1}

    print(key_chem)
    print(set(utils.CHEMICAL_COLS) - set(key_chem))

    df = data_refinement.load_geo_chem_data()
    data_visualizer.show_lda_plot(df, key_chem.keys(), ['Heiberg', 'Villard', 'Ivanhoe', 'Moland', 'Twin Cities', 'Verdi', 'Dovray'])

def hyper_train():
    config = {
        'n_estimators' : tune.randint(50, 500),
        'max_depth' : tune.randint(4, 10),
        'noramlize_type' : tune.choice(['tree', 'forest']),
        'sample_type' : tune.choice(['uniform', 'weighted']),
        'learning_rate' : tune.loguniform(1e-4, 1e-1),
        'subsample' : tune.uniform(.5, 1.0)
    }

    tuner = tune.Tuner(
        xgb_model_trainer.hyperparameter_train_classifier,
        tune_config=tune.TuneConfig(
            mode='max',
            metric='accuracy',
            search_alg=OptunaSearch(),
            num_samples=10,
        ),
        param_space=config
    )

    results = tuner.fit()

    print(f'Best config: {results.get_best_result().config}')

df = data_refinement.load_geo_chem_data()
data_visualizer.show_lda_biplot(df, utils.CHEMICAL_COLS)