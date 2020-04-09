# sys.path
import sys
mlibpath = r"C:\Users\m038402\Documents\myWork\pythoncodes\mlib"
sys.path.append(mlibpath)

from utility import load_csv, load_td, connect_td 
from categorize import Categorizer

data_file = "./data/data.csv"

df = load_csv(data_file)
df.head()

# -------------------------------------------------------------------
# categorizer
bprofile = Categorizer(df, "TARGET_F")

sdf = df.copy()
# mutable...
bprofile.categorize(sdf)
sdf.head()
df.head()
bprofile.get_df_column_type()
bprofile.get_likelihood("CUNCZZ_AGE_YEARS", False)

bprofile = Categorizer(df, "TARGET_F")
bprofile.df.head()
bprofile.get_crosstab("CUNCZZ_AGE_YEARS", False)
# bprofile.unique_values
# bprofile.num_bins

colname = "CUNCZZ_AGE_YEARS"
# -------------------------------------------------------------------
# profiler

from varprofile import Profiler 
profiler = Profiler(df, 'TARGET_F')
profiler.df.head()
profiler.df[colname].head()
rank = profiler.iv_ranking()
rank.iv[0]


profiler.count_unique_values()
profiler.get_categorical_column()
profiler.get_woe(colname, categorical=False)
profiler.get_crosstab(colname, categorical=False)
profiler.get_iv(colname, categorical=False)
profiler.get_profile(colname, categorical=False)
