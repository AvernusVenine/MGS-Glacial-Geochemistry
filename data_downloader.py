import arcpy
import pandas as pd
import numpy as np

import data_refinement
import utils
from utils import Field

# Loads data from the QDI dataset
def download_qdi_data():
    ix_numpy_array = arcpy.da.FeatureClassToNumPyArray(
        f'{utils.SDE_CONN}\\{utils.QDI_IX_DATABASE_PATH}',
        field_names=utils.IX_FIELDS,
        skip_nulls=False,
        null_value=0
    )

    ix_df = pd.DataFrame(data=ix_numpy_array)
    ix_df = ix_df.drop_duplicates(subset=[Field.RELATEID])

    tx_numpy_array = arcpy.da.FeatureClassToNumPyArray(
        f'{utils.SDE_CONN}\\{utils.QDI_TX_DATABASE_PATH}',
        field_names=utils.TX_FIELDS,
        skip_nulls=False,
        null_value=0
    )
    tx_df = pd.DataFrame(data=tx_numpy_array)

    tx_df[Field.ELEVATION] = tx_df[Field.RELATEID].map(ix_df.set_index(Field.RELATEID)[Field.ELEVATION])
    tx_df[Field.UTME] = tx_df[Field.RELATEID].map(ix_df.set_index(Field.RELATEID)[Field.UTME])
    tx_df[Field.UTMN] = tx_df[Field.RELATEID].map(ix_df.set_index(Field.RELATEID)[Field.UTMN])

    tx_df = tx_df.replace({'0' : None, '' : None})
    tx_df = tx_df.dropna(subset=[Field.UNIT, Field.UTME, Field.UTMN])
    tx_df = tx_df.fillna(value=0)

    value_counts = tx_df[Field.UNIT].value_counts()
    mask = value_counts[value_counts < utils.MIN_COUNT].index
    tx_df.loc[tx_df[Field.UNIT].isin(mask), Field.UNIT] = 'Undifferentiated'

    #tx_df = data_refinement.scale_df(tx_df, utils.SCALED_COLS)

    tx_df.to_csv(utils.QDI_SAMPLE_PATH, index=False)

# TODO: Finish this
# Finds the map label at a given sample point and saves it to the dataframe
def download_map_data(df : pd.DataFrame):
    layer = arcpy.mp.LayerFile(utils.QDI_MAP_PATH)
    layer = layer.listLayers()[0]

    spatial_ref = arcpy.Describe(layer).spatialReference

    arcpy.management.MakeFeatureLayer(layer, 'temp')

    for _, row in df.iterrows():
        point = arcpy.PointGeometry(arcpy.Point(row[Field.UTME], row[Field.UTMN]), spatial_ref)

# Converts the geo chem data found in an excel file into a readable csv
def download_geo_chem_data(df : pd.DataFrame):
    gc_df = pd.read_excel(
        utils.GEO_CHEM_EXCEL_PATH,
        sheet_name='Sheet1',
        na_values=0,
        )

    gc_df = gc_df.replace({' ' : 0})

    gc_df = gc_df.drop(columns=['Recvd Wt.'])

    gc_df[utils.CHEMICAL_COLS + [Field.DEPTH]] = gc_df[utils.CHEMICAL_COLS + [Field.DEPTH]].astype(float)

    gc_df.to_csv(utils.GEO_CHEM_PATH, index=False)

def get_data_value_counts():
    tx_numpy_array = arcpy.da.FeatureClassToNumPyArray(
        f'{utils.SDE_CONN}\\{utils.QDI_TX_DATABASE_PATH}',
        field_names=utils.TX_FIELDS,
        skip_nulls=False,
        null_value=0
    )
    tx_df = pd.DataFrame(data=tx_numpy_array)

    print(tx_df[Field.UNIT].value_counts())

download_qdi_data()