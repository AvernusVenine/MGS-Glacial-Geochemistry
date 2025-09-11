class Field:
    RELATEID = 'relateid'

    UTME = 'utme' #from IX
    UTMN = 'utmn' #from IX
    SAMPLE_DEPTH = 'sampledepth'
    ELEVATION = 'elevation' #from IX

    UNIT = 'stratunit'

    SAND_PERCENTAGE = 'sand_pct'
    SILT_PERCENTAGE = 'silt_pct'
    CLAY_PERCENTAGE = 'clay_pct'

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

    MAP_LABEL = 'maplabel'
    SHAPE_XY = 'SHAPE@XY'

    SAMPLE_NUM = 'sample_num'
    INTERPRETATION = 'interpretation'
    DEPTH = 'depth'

    AG_PPM = 'Ag_ppm'
    AL_PERCENTAGE = 'Al_%'
    AS_PPM = 'As_ppm'
    BA_PPM = 'Ba_ppm'
    BE_PPM = 'Be_ppm'
    BI_PPM = 'Bi_ppm'
    CA_PERCENTAGE = 'Ca_%'
    CD_PPM = 'Cd_ppm'
    CE_PPM = 'Ce_ppm'
    CO_PPM = 'Co_ppm'
    CR_PPM = 'Cr_ppm'
    CS_PPM = 'Cs_ppm'
    CU_PPM = 'Cu_ppm'
    FE_PERCENTAGE = 'Fe_%'
    GA_PPM = 'Ga_ppm'
    GE_PPM = 'Ge_ppm'
    HF_PPM = 'Hf_ppm'
    IN_PPM = 'In_ppm'
    K_PERCENTAGE = 'K_%'
    LA_PPM = 'La_ppm'
    LI_PPM = 'Li_ppm'
    MG_PERCENTAGE = 'Mg_%'
    MN_PPM = 'Mn_ppm'
    MO_PPM = 'Mo_ppm'
    NA_PERCENTAGE = 'Na_%'
    NB_PPM = 'Nb_ppm'
    NI_PPM = 'Ni_ppm'
    P_PPM = 'P_ppm'
    PB_PPM = 'Pb_ppm'
    RB_PPM = 'Rb_ppm'
    RE_PPM = 'Re_ppm'
    S_PERCENTAGE = 'S_%'
    SB_PPM = 'Sb_ppm'
    SC_PPM = 'Sc_ppm'
    SE_PPM = 'Se_ppm'
    SN_PPM = 'Sn_ppm'
    SR_PPM = 'Sr_ppm'
    TA_PPM = 'Ta_ppm'
    TE_PPM = 'Te_ppm'
    TH_PPM = 'Th_ppm'
    TI_PERCENTAGE = 'Ti_%'
    TL_PPM = 'Tl_ppm'
    U_PPM = 'U_ppm'
    V_PPM = 'V_ppm'
    W_PPM = 'W_ppm'
    Y_PPM = 'Y_ppm'
    ZN_PPM = 'Zn_ppm'
    ZR_PPM = 'Zr_ppm'

FORMATION_MAP = {
    'Heiberg' : 'New Ulm',
    'Meyer Lake' : 'Lake Henry',
    'Villard' : 'New Ulm',
    'Sauk Centre' : 'Lake Henry',
    'GT3' : 'Good Thunder',
    'Independence, South Long Lake' : 'Independence',
    'Mille Lacs' : 'Cromwell',
    'Automba' : 'Cromwell',
    'Emerald' : 'Cromwell',
    'Cromwell, Mille Lacs, Automba or St. Croix' : 'Cromwell',
    'St. Croix' : 'Cromwell',
    'Independence, proto Brainerd lobe' : 'Independence',
    'Independence, Proto-Brainerd lobe' : 'Independence',
    'GT1' : 'Good Thunder',
    'GT2' : 'Good Thunder',
    'GT4' : 'Good Thunder',
    'GT5' : 'Good Thunder',
    'Dovray' : 'New Ulm',
    'Ivanhoe' : 'New Ulm',
    'Moland' : 'New Ulm',
    'Twin Cities' : 'New Ulm',
    'Verdi' : 'New Ulm',
    'Moose Lake' : 'Barnum',
    'Boundary Waters' : 'Boundary Waters',
    'Browerville' : 'Browerville',
    'Independence' : 'Independence',
    'Hewitt' : 'Hewitt',
    'St. Francis' : 'St. Francis',
    'St, Francis, Upper' : 'St. Francis',
    'St. Francis, Lower' : 'St. Francis',
    'Alborn' : 'Aitkin',
    'Nelson Lake' : 'Aitkin',
    'Prairie Lake' : 'Aitkin',
}

SDE_CONN = 'PostgreSQL-134-mgs_qdi(mgsstaff).sde'
QDI_TX_DATABASE_PATH = 'mgs_qdi.qdi.qdtx' # Table that contains info about layer contents
QDI_IX_DATABASE_PATH = 'mgs_qdi.qdi.qdix' # Table that contains info about sample site
QDI_MAP_PATH = 'data/glacial_map.lyr'
GEO_CHEM_EXCEL_PATH = 'data/geochem_data.xlsx'

QDI_SAMPLE_PATH = 'data/samples.csv'
GEO_CHEM_PATH = 'data/geo_chem.csv'

MIN_COUNT = 25

MIN_LITH_COLS = [
    Field.CRYSTALLINE_PERCENTAGE,
    Field.CARBONATE_PERCENTAGE,
    Field.SHALE_PERCENTAGE,
    Field.PRECAMBRIAN_PERCENTAGE,
    Field.PALEOZOIC_PERCENTAGE,
    Field.CRETACEOUS_PERCENTAGE,
    Field.LIGHT_PERCENTAGE,
    Field.DARK_PERCENTAGE,
    Field.RED_PERCENTAGE,
]

LITHOLOGY_COLS = [
    Field.SAND_PERCENTAGE,
    Field.SILT_PERCENTAGE,
    Field.CLAY_PERCENTAGE,
    Field.CRYSTALLINE_PERCENTAGE,
    Field.CARBONATE_PERCENTAGE,
    Field.SHALE_PERCENTAGE,
    Field.PRECAMBRIAN_PERCENTAGE,
    Field.PALEOZOIC_PERCENTAGE,
    Field.CRETACEOUS_PERCENTAGE,
    Field.LIGHT_PERCENTAGE,
    Field.DARK_PERCENTAGE,
    Field.RED_PERCENTAGE,
]

CHEMICAL_COLS = [
    Field.AG_PPM,
    Field.AL_PERCENTAGE,
    Field.AS_PPM,
    Field.BA_PPM,
    Field.BE_PPM,
    Field.BI_PPM,
    Field.CA_PERCENTAGE,
    Field.CD_PPM,
    Field.CE_PPM,
    Field.CO_PPM,
    Field.CR_PPM,
    Field.CS_PPM,
    Field.CU_PPM,
    Field.FE_PERCENTAGE,
    Field.GA_PPM,
    Field.GE_PPM,
    Field.HF_PPM,
    Field.IN_PPM,
    Field.K_PERCENTAGE,
    Field.LA_PPM,
    Field.LI_PPM,
    Field.MG_PERCENTAGE,
    Field.MN_PPM,
    Field.MO_PPM,
    Field.NA_PERCENTAGE,
    Field.NB_PPM,
    Field.NI_PPM,
    Field.P_PPM,
    Field.PB_PPM,
    Field.RB_PPM,
    Field.RE_PPM,
    Field.S_PERCENTAGE,
    Field.SB_PPM,
    Field.SC_PPM,
    Field.SE_PPM,
    Field.SN_PPM,
    Field.SR_PPM,
    Field.TA_PPM,
    Field.TE_PPM,
    Field.TH_PPM,
    Field.TI_PERCENTAGE,
    Field.TL_PPM,
    Field.U_PPM,
    Field.V_PPM,
    Field.W_PPM,
    Field.Y_PPM,
    Field.ZN_PPM,
    Field.ZR_PPM
]

SCALED_COLS = [
    Field.UTME,
    Field.UTMN,
    Field.SAMPLE_DEPTH,
    Field.ELEVATION
]

TX_FIELDS = [
    Field.RELATEID,
    Field.SAMPLE_NUM,
    Field.UNIT,
    Field.SAMPLE_DEPTH,
    Field.SAND_PERCENTAGE,
    Field.SILT_PERCENTAGE,
    Field.CLAY_PERCENTAGE,
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