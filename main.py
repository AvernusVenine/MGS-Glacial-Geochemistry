import pandas
from xgboost import plot_importance
import matplotlib.pyplot as plt
import data_refinement
import utils
import xgb_model_trainer
from data_refinement import Field
import data_downloader

def train_geo_chem():
    df = data_refinement.load_geo_chem_data()
    old_df = df.copy()
    value_counts = df[Field.UNIT].value_counts()

    mask = value_counts[value_counts < 10].index
    df.loc[df[Field.UNIT].isin(mask), Field.UNIT] = 'Undifferentiated'

    df, encoder = data_refinement.label_encode_units(df)

    model = xgb_model_trainer.train_geo_chem_model_binary(df, encoder, ['QINU'])
    #model = xgb_model_trainer.train_geo_chem_model(df, encoder)
    old_df = old_df.dropna()
    data_refinement.show_pca_plot(old_df, xgb_model_trainer.get_unused_features(model), 'QBVU')

    plot_importance(model, importance_type='gain')
    plt.show()

#train_geo_chem()

df = data_refinement.load_geo_chem_data().dropna()
data_refinement.show_pca_plot(df, utils.CHEMICAL_COLS, 'QINU')
#data_refinement.show_3d_plot(df, [Field.TH_PPM, Field.MO_PPM, Field.BE_PPM], 'QBVU')
