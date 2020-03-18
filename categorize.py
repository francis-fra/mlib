import pandas as pd
import numpy as np
import sys, os
# import pyodbc, getpass
from operator import itemgetter
import argparse
# from utility import load_csv, load_td, connect_td 

class Categorizer(object):
    def __init__(self, df, target_col, num_bins=5, savefile=None):

        assert num_bins > 1 and num_bins <= 9, "numbins must be greater than 1 and less than 10"
        self.target_col = target_col
        self.num_bins = num_bins
        self.df = df
        self.savefile = savefile
        # all columns are in uppercase
        self.col_uppercase()
        self.unique_values = self.count_unique_values()

    def count_unique_values(self, df=None, prop=False, subset=None, dropna=True):
        """Return a dictionary {column name, num_unique_values}"""
        if df is None:
            df = self.df
        if subset is None:
            subset = df.columns
        if (isinstance(subset, str)):
            subset = [subset]
        if prop == False:
            f = lambda x: x.nunique(dropna)
        else:
            f = lambda x: x.nunique(dropna) / df.shape[0]
        return (df[subset].T.apply(f, axis=1).to_dict())

    def get_crosstab(self, colname, categorical=False):
        '''
            profile one single variable
            Calculates the cross table or KL divergence

            :param: var_name  : column name prefix in resultant data frame
            :param: raw_count : cross table (if True), KL diverence otherwise
            :param: categorical: True if colname is a categorical variable
        '''
        df = self.df
        num_bins = self.num_bins
        target_col = self.target_col
        num_unique = self.unique_values
        # num_unique = self.count_unique_values(df, subset=target_col)
        if not categorical:
            tmp = pd.qcut(df[colname], num_bins, retbins=True, duplicates='drop')
            profile = pd.crosstab(df[target_col], tmp[0])
            # rename binned columns
            # newcolname = [var_name + '_' + str(v) for v in range(profile.shape[1])]
            maxval = profile.shape[1]
            newcolname = self.get_category_name(maxval)
            profile.columns = newcolname
        else:
            num_unique = self.count_unique_values(df, subset=colname)
            num_bins = num_unique[colname]
            profile = pd.crosstab(df[target_col], df[colname])
        return profile

    def get_likelihood(self, colname, categorical=False):
        "get the likelihood reponse rate for a feature"

        profile = self.get_crosstab(colname, categorical)
        total = sum(profile.apply(lambda x: sum(x)))
        total_target = sum(profile.iloc[1])
        base_rate = total_target / total

        # total count at each group
        grp_total = profile.apply(lambda x: sum(x))
        grp_target_count = profile.iloc[1]
        grp_rate = grp_target_count / grp_total
        # series
        likelihood = grp_rate / base_rate - 1
        return pd.DataFrame({colname: likelihood}).T

    # data frame utilities
    def count_unique_values(self, df=None, prop=False, subset=None, dropna=True):
        """Return a dictionary {column name, num_unique_values}"""

        if df is None:
            df = self.df
        if subset is None:
            subset = df.columns
        if (isinstance(subset, str)):
            subset = [subset]
        if prop == False:
            f = lambda x: x.nunique(dropna)
        else:
            f = lambda x: x.nunique(dropna) / df.shape[0]
        return (df[subset].T.apply(f, axis=1).to_dict())


    def get_df_column_type(self, df=None, exclusions=None):
        "identify column type based on MDM naming scheme"
        def get_column_type(colname):
            """
                'B': binary, 'C': categorical, 'N': numeric
            """
            return colname[2].upper()

        if df is None:
            df = self.df
        if exclusions is None:
            exclusions = []
        cols = list(set(df.columns) - set(exclusions))
        col_type = [get_column_type(x) for x in cols]
        col_dict = dict(zip(cols, col_type))

        unique_types = {'Binary': 'B', 'Categorical': 'C', 'Numerical': 'N'}
        typeDict = {}
        for key in unique_types.keys():
            typeDict[key] = [c for c in cols if col_dict[c] == unique_types[key]]

        return typeDict

    def col_uppercase(self):
        self.df.columns = self.df.columns.map(lambda x: x.upper())

    # mutable operation
    def categorize(self, df=None, exclusions=None):
        "bin values in all columns"

        def replace_value(series, colname):
            count = 0
            for col in colname:
                series = series.replace(count, col)
                count += 1
            return series

        def replace_na(series, fill_value):
            series.fillna(fill_value)
            return series

        if df is None:
            df = self.df.copy()
        if exclusions is None:
            exclusions = []

        num_bins = self.num_bins
        type_dict = self.get_df_column_type(df, exclusions=exclusions)

        numerical_columns = type_dict['Numerical']
        binary_columns = type_dict['Binary']
        for col in numerical_columns:
            df[col] = pd.qcut(df[col], num_bins, duplicates='drop', labels=False)
            # num unique values
            maxval = np.nanmax(df[col]) + 1
            labels = self.get_category_name(maxval)
            df[col] = replace_value(df[col], labels)
            df[col] = replace_na(df[col], 0.0)
        for col in binary_columns:
            df[col] = replace_value(df[col], ['No', 'Yes'])
            df[col] = replace_na(df[col], 'NA')

        # fill all other (categorical) missing values
        df = df.fillna(value='NA')

        return df

    def get_category_name(self, maxval):
        category_name = ['lowest', 'very low', 'lower', 'low', 'mid', 'high', 'higher', 'very high', 'highest']
        if maxval == 2:
            return itemgetter(3, 5)(category_name)
        elif maxval == 3:
            return itemgetter(3, 4, 5)(category_name)
        elif maxval == 4:
            return itemgetter(1, 2, 5, 6)(category_name)
        elif maxval == 5:
            return itemgetter(2, 3, 4, 5, 6)(category_name)
        elif maxval == 6:
            return itemgetter(1, 2, 3, 5, 6, 7)(category_name)
        elif maxval == 7:
            return itemgetter(1, 2, 3, 4, 5, 6, 7)(category_name)
        elif maxval == 8:
            return itemgetter(0, 1, 2, 3, 5, 6, 7, 8)(category_name)
        else:
            return tuple(category_name)

if __name__ == "__main__":
    from utility import load_csv, load_td, connect_td 
    parser = argparse.ArgumentParser()
    # positional
    parser.add_argument("table", type=str, help="data source table name")
    # optional
    # parser.add_argument("-x", "--crosstab", type=str, help="crosstab output")
    parser.add_argument("-x", "--exclude", type=str, default=['CUSTOMER_ID', 'GCIS_KEY'], nargs='+', help="columns to be excluded")
    parser.add_argument("-r", "--rows", type=int, help="number of rows to select")
    parser.add_argument("-n", "--bins", type=int, help="number of bins to select")
    parser.add_argument("-p", "--profile", type=str, help="profile column")
    parser.add_argument("-t", "--target", type=str, default='TARGET_F', help="target column name")
    parser.add_argument("-o", "--out", type=str, help="output file")
    args = parser.parse_args()

    # identify csv / teradata
    iscsv = args.table.endswith(".csv")

    # read csv / teradata
    if iscsv:
        try:
            df = load_csv(args.table, args.rows)
        except:
            print("Oops!  Data table not found...")
            sys.exit
    else:
        try:
            cnxn = connect_td()
        except:
            print("Oops!  Fail to connect to TD...")
            sys.exit
        try:
            df = load_td(args.table, cnxn, args.rows)
        except:
            print("Oops!  Data table not found...")
            sys.exit

    # -----------------------------------------------------------------------------
    # set up Categorizer
    # -----------------------------------------------------------------------------
    if args.bins is not None:
        bprofile = Categorizer(df, args.target, args.bins)
    else:
        bprofile = Categorizer(df, args.target)


    if args.profile is None:
        df = bprofile.categorize(exclusions=args.exclude)
    else:
        type_dict = bprofile.get_df_column_type(df, exclusions=args.exclude)
        numerical_columns = type_dict['Numerical']
        is_numeric = args.profile in numerical_columns
        df = bprofile.get_likelihood(args.profile, not is_numeric)

    if args.out is None:
        df.to_csv(sys.stdout)
    else:
        df.to_csv(args.out, index=False)
