import arcpy
import pandas
import data_refinement
from data_refinement import Field

df = data_refinement.load_data()

value_counts = df[Field.UNIT].value_counts()
value_counts = value_counts[value_counts > 25]

print(value_counts)