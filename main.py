import pandas
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
import data_refinement
import utils
import xgb_model_trainer
from data_refinement import Field
import data_downloader


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
    data_refinement.show_lda_plot(df, key_chem.keys(), ['Heiberg', 'Villard', 'Ivanhoe', 'Moland', 'Twin Cities', 'Verdi', 'Dovray'])

#find_key_chemicals()

pd.option_context('display.max_rows', None,
                       'display.max_columns', 48,
                       'display.precision', 2,
                       )

df = data_refinement.load_geo_chem_data()

df = df[df[Field.INTERPRETATION].isin(['Heiberg', 'Villard', 'Ivanhoe', 'Moland', 'Twin Cities', 'Verdi', 'Dovray'])]

ivanhoe = df[df[Field.INTERPRETATION] == 'Ivanhoe']
other = df[df[Field.INTERPRETATION] != 'Ivanhoe']

ivanhoe = ivanhoe[utils.CHEMICAL_COLS]
other = other[utils.CHEMICAL_COLS]

print(pd.DataFrame(data=[ivanhoe.mean(), ivanhoe.std()]))
print(pd.DataFrame(data=[other.mean(), other.std()]))

#df = df.replace(utils.FORMATION_MAP)
#data_refinement.show_lda_plot(df, utils.CHEMICAL_COLS, ['Boundary Waters', 'Saum', 'Hewitt', 'Independence'], False)

#data_refinement.show_pca_plot(df, utils.CHEMICAL_COLS, ['Heiberg', 'New Ulm'], True)
#train_geo_chem()

#data_refinement.show_correlation_matrix(df)
#data_refinement.show_3d_plot(df, [Field.IN_PPM, Field.NB_PPM, Field.GA_PPM], 'Browerville')
