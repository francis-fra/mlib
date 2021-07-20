import utility as ut
import explore as ex 
from categorize import Categorizer

data_file = "../data/data.csv"

df = load_csv(data_file)
df.head()

# ------------------------------------------------------------------------
# utility
# ------------------------------------------------------------------------
tmp = ut.widedf_to_talldf(df, id_vars=["TARGET_F"])
tmp.head()
tmp.shape
df.shape

ex.count_unique_values(df)
ex.count_unique_values(tmp)

(X, Y, feature) = ut.extract_feature_target(df, "TARGET_F")
X
Y
feature

ut.get_unique_values(df, "TARGET_F")

# ------------------------------------------------------------------------
# explore
# ------------------------------------------------------------------------
ex.get_type_dict(df)
colname = "CUNCZZ_AGE_YEARS"
ex.get_column_type(df, colname)
ex.get_distinct_value(df, colname)
ex.get_type_tuple(df)
ex.get_categorical_column(df)
ex.get_non_categorical_column(df)


from imp import reload
reload(ex)
ex.get_column_type(df)

ex.count_unique_values(df)
# get distribution of values
ex.count_levels(df, colname)
ex.count_missing(df)