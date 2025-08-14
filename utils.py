class Field:
    RELATEID = 'relateid'

    UTME = 'utme' #from IX
    UTMN = 'utmn' #from IX
    SAMPLE_DEPTH = 'sampledepth'
    ELEVATION = 'elevation' #from IX

    UNIT = 'stratunit'

    SAND_PERCENTAGE = 'sand_pct'
    SILT_PERCENTAGE = 'silt_pct'
    ClAY_PERCENTAGE = 'clay_pct'

    CRYSTALLINE_PERCENTAGE = 'crystalline_blk'
    CARBONATE_PERCENTAGE = 'carbonate_blk'
    SHALE_PERCENTAGE = 'shale_blk'
    GRAY_SHALE_PERCENTAGE = 'gray_cls'

    PRECAMBRIAN_PERCENTAGE = 'precambrian_blk'
    PALEOZOIC_PERCENTAGE = 'paleozoic_blk'
    CRETACEOUS_PERCENTAGE = 'cretaceous_blk'
    LIMESTONE_PERCENTAGE = 'lms_cls'

    LIGHT_PERCENTAGE = 'light_cls'
    DARK_PERCENTAGE = 'dark_cls'
    RED_PERCENTAGE = 'red_cls'
    CLEAR_PERCENTAGE = 'clear_quartz_cls'

SDE_CONN = 'PostgreSQL-134-mgs_qdi(mgsstaff).sde'
QDI_TX_DATA_PATH = 'mgs_qdi.qdi.qdtx' # Table that contains info about layer contents
QDI_IX_DATA_PATH = 'mgs_qdi.qdi.qdix' # Table that contains info about sample site

TX_FIELDS = [
    Field.RELATEID,
    Field.UNIT,
    Field.SAMPLE_DEPTH,
    Field.SAND_PERCENTAGE,
    Field.SILT_PERCENTAGE,
    Field.ClAY_PERCENTAGE,
    Field.CRYSTALLINE_PERCENTAGE,
    Field.CARBONATE_PERCENTAGE,
    Field.SHALE_PERCENTAGE,
    Field.GRAY_SHALE_PERCENTAGE,
    Field.PRECAMBRIAN_PERCENTAGE,
    Field.PALEOZOIC_PERCENTAGE,
    Field.CRETACEOUS_PERCENTAGE,
    Field.LIMESTONE_PERCENTAGE,
    Field.LIGHT_PERCENTAGE,
    Field.DARK_PERCENTAGE,
    Field.RED_PERCENTAGE,
    Field.CLEAR_PERCENTAGE,
]
IX_FIELDS = [
    Field.RELATEID,
    Field.ELEVATION,
    Field.UTME,
    Field.UTMN,
]