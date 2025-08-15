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

    value_counts = df[Field.UNIT].value_counts()

    mask = value_counts[value_counts < 10].index
    df.loc[df[Field.UNIT].isin(mask), Field.UNIT] = 'Undifferentiated'

    df, encoder = data_refinement.label_encode_units(df)

    # model = xgb_model_trainer.train_geo_chem_model_binary(df, encoder, ['QELU'])
    model = xgb_model_trainer.train_geo_chem_model(df, encoder)
    plot_importance(model, importance_type='weight')
    plt.show()

df = data_refinement.load_geo_chem_data()
df = df[utils.CHEMICAL_COLS]

cm = df.corr()

plt.imshow(cm, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.xticks(range(len(cm)), cm.columns, rotation=45, ha='right')
plt.yticks(range(len(cm)), cm.columns)
plt.tight_layout()
plt.show()