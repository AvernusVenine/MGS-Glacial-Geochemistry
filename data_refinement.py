import pandas as pd
import arcpy

import utils
from utils import Field

def load_data():
    ix_numpy_array = arcpy.da.FeatureClassToNumPyArray(
        f'{utils.SDE_CONN}\\{utils.QDI_IX_DATA_PATH}',
        field_names=utils.IX_FIELDS,
        skip_nulls=False,
        null_value=0
    )
    ix_df = pd.DataFrame(data=ix_numpy_array)
    ix_df = ix_df.drop_duplicates(subset=[Field.RELATEID])

    tx_numpy_array = arcpy.da.FeatureClassToNumPyArray(
        f'{utils.SDE_CONN}\\{utils.QDI_TX_DATA_PATH}',
        field_names=utils.TX_FIELDS,
        skip_nulls=False,
        null_value=0
    )
    tx_df = pd.DataFrame(data=tx_numpy_array)

    tx_df = tx_df.replace({'0' : None, '' : None})
    tx_df = tx_df.dropna(subset=[Field.UNIT])

    tx_df[Field.ELEVATION] = tx_df[Field.RELATEID].map(ix_df.set_index(Field.RELATEID)[Field.ELEVATION])
    tx_df[Field.UTME] = tx_df[Field.RELATEID].map(ix_df.set_index(Field.RELATEID)[Field.UTME])
    tx_df[Field.UTMN] = tx_df[Field.RELATEID].map(ix_df.set_index(Field.RELATEID)[Field.UTMN])

    return tx_df

def map_units(df : pd.DataFrame):



    pass