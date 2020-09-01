#!python

import pandas as pd
import numpy as np
import sys

import argparse


class Profiler(object):
    def __init__(self, df, target_col, num_bins=10, savefile=None):
        self.target_col = target_col
        self.num_bins = num_bins
        self.df = df
        self.savefile = savefile
        self.num_target_class = self.__get_num_target_class()
        if (self.num_target_class) <= 1:
            raise Exception("Invalid Target")

    def __get_num_target_class(self):
        "get num target class"
        num_unique = self.count_unique_values(self.df, subset=self.target_col)
        return num_unique[self.target_col]

    def iv_ranking(self, columns=None):
        "calculate IV values and rank"

        df = self.df
        target_col = self.target_col
        num_bins = self.num_bins

        # find categorical columns
        categorical_cols = self.get_categorical_column()

        if columns is None:
            columns = df.columns

        columns = list(columns)
        columns.remove(target_col)
        result = pd.DataFrame({'iv': np.zeros(len(columns))}, index=columns)

        for colname in columns:
            is_categorical = colname in categorical_cols
            try:
                result.loc[colname] = self.get_iv(colname, categorical=is_categorical)
            except:
                print('ERROR: skipping {}'.format(colname))

        result = result.sort_values(by=['iv'], ascending=False)

        return result

    def get_iv(self, colname, var_name='GRP', categorical=False, weighted=False):
        "calculate IV values"

        (woe, targetDst, altDst) = self.get_woe(colname, var_name, categorical, distOut=True)

        # check if this is multi class
        ismulti = len(targetDst.shape) == 2
        num_bins = len(altDst)

        # multi-class
        if ismulti:
            for k in range(num_bins):
               targetDst.iloc[:,k] = (targetDst.iloc[:,k] - altDst[k] ) * woe.iloc[:,k]

            if weighted == True:
                # TODO: weighted IV
                pass
            else:
                # find the global sum if multi class
                iv = targetDst.sum().sum()
        else:
            # binary class
            iv = sum((targetDst - altDst) * woe)

        return iv


    def get_woe(self, colname, var_name='GRP', categorical=False, distOut=False):
        '''calculate WoE

            Parameter
            ---------
            :param: colname   : columne to crosstab with the target
            :param: var_name  : column name prefix in resultant data frame
            :param: categorical: True if colname is a categorical variable

            Returns
            -------
            woe for each possible value against the target

        '''

        # get the distribution
        (targetDst, altDst) = self.get_distribution(colname, var_name, categorical)
        woe = targetDst.copy()

        # check if this is multi class
        ismulti = len(targetDst.shape) == 2
        num_bins = len(altDst)

        # get the woe for all possible target value with all possible value in column
        # e.g. for binary : 0 or 1
        # multi-class
        if ismulti:
            np.seterr(divide='ignore')
            for k in range(num_bins):
                # replace any inf with 0 from the series 
                likelihood = np.log(targetDst.iloc[:,k] / altDst[k])
                likelihood = likelihood.replace([-np.inf, np.inf], 0)
                woe.iloc[:,k] = likelihood
        else:
            for k in range(num_bins):
                # column selection
                likelihood = np.log(targetDst.iloc[k] / altDst[k])
                woe.iloc[k] = 0 if np.abs(likelihood) == np.inf else likelihood

        if distOut:
            return (woe, targetDst, altDst)
        else:
            return woe

    def get_crosstab(self, colname, var_name='GRP', categorical=False):
        '''
            profile one single variable
            Calculates the cross table

            Parameter
            ---------
            :param: colname   : columne to crosstab with the target
            :param: var_name  : column name prefix in resultant data frame
            :param: categorical: True if colname is a categorical variable

            Returns
            -------
            crosstab table

        '''
        df = self.df
        num_bins = self.num_bins
        target_col = self.target_col

        if not categorical:
            tmp = pd.qcut(df[colname], num_bins, retbins=True, duplicates='drop')
            profile = pd.crosstab(df[target_col], tmp[0])
            # rename binned columns
            newcolname = [var_name + '_' + str(v) for v in range(profile.shape[1])]
            profile.columns = newcolname
        else:
            # num_unique = self.count_unique_values(df, subset=colname)
            # num_bins = num_unique[colname]
            profile = pd.crosstab(df[target_col], df[colname])

        return profile


    def get_distribution(self, colname, var_name='GRP', categorical=False):
        '''

            For binary target, get target and no-target distribution
            For multiclass, the alternative distribution is the global distribution

            Parameter
            ---------
            :param: colname   : columne to crosstab with the target
            :param: var_name  : column name prefix in resultant data frame
            :param: categorical: True if colname is a categorical variable

            Returns
            -------
            targetDst: target distribution
            altDst : non-target distribution (for binary class) or
                    global distribution (for multi class)

        '''

        # get cross tab
        profile = self.get_crosstab(colname, var_name, categorical)

        num_segments = self.num_target_class

        # WoE for multi-class
        if num_segments > 2:
            # overall distribution
            bin_distribution = profile.sum() / sum(profile.sum())

            # calculate likelihood
            for k in range(num_segments):
                # row selection
                profile.iloc[k] = profile.iloc[k] / sum(profile.iloc[k])

            altDst = bin_distribution
            targetDst = profile

        # elif num_segments == 2:
        else:

            for k in range(num_segments):
                # row selection
                profile.iloc[k] = profile.iloc[k] / sum(profile.iloc[k])

            altDst = profile.iloc[0]
            targetDst = profile.iloc[1]

        # else:
        #     print('ERROR: Number of target is too less')
        #     altDst = None
        #     targetDst = None

        # return target distribution and alternative distribution
        return (targetDst, altDst)



    # def widedf_to_talldf(self, df, id_vars=None):
    #     "convert wide to tall data frame"

    #     if id_vars is None:
    #         id_vars = self.target_col
    #     df.reset_index(level=0, inplace=True)
    #     df = pd.melt(df, id_vars=[id_vars])

    #     return df

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

    def get_categorical_column(self, df=None, exclusions=None, index=False, max_distinct=15, cutoff = 0.01):
        """
            Return [list of categorical column]

            Categorical column is either:
            1) non-numerical
            2) numeric but small number of finite values

        """
        if df is None:
            df = self.df

        if exclusions is None:
            exclusions = []

        if (isinstance(exclusions, str)):
            exclusions = [exclusions]

        result = []
        result += self.get_non_numerical_column(df)

        # dictionary of unique values proportion
        d = self.count_unique_values(df, prop=True)
        c = self.count_unique_values(df, prop=False)

        small_prop_set = [k for (k, v) in d.items() if v < cutoff]
        small_finite_set = [k for (k, v) in c.items() if v < max_distinct]

        # AND condition
        result += list(set(small_prop_set) & set(small_finite_set))

        result = list(set(result) - set(exclusions))

        if index == False:
            return (result)
        else:
            return [df.columns.get_loc(x) for x in result]


    def get_non_numerical_column(self, df=None, index=False):
        """Return [list of numerical column]"""

        if df is None:
            df = self.df

        cols = list(set(df.columns) - set(df._get_numeric_data().columns.tolist()))
        if index == True:
            return [df.columns.get_loc(x) for x in cols]
        else:
            return (cols)

#------------------------------------------------------------------------
# python xxx.py abc.csv -c var -r rows
# python xxx.py c4ustpmk.nw_cust_m_hist -c var -r rows
# python varprofile.py profile.csv -x CLNMZZ_Floor_Area

if __name__ == "__main__":
    from utility import load_csv, load_td, connect_td 
    parser = argparse.ArgumentParser()
    # positional
    parser.add_argument("table", type=str, help="data source table name")
    # optional
    parser.add_argument("-x", "--crosstab", type=str, help="crosstab output")
    parser.add_argument("-c", "--column", type=str, help="column name in the table")
    parser.add_argument("-r", "--row", type=int, help="number of rows to select")
    parser.add_argument("-n", "--bins", type=int, help="number of bins to select")
    args = parser.parse_args()

    # identify csv / teradata
    iscsv = args.table.endswith(".csv")

    # read csv / teradata
    if iscsv:
        try:
            df = load_csv(args.table, args.row)
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
            df = load_td(args.table, cnxn, args.row)
        except:
            print("Oops!  Data table not found...")
            sys.exit

    # set up profiler
    if args.bins is not None:
        bprofile = Profiler(df, 'TARGET_F', args.bins)
    else:
        bprofile = Profiler(df, 'TARGET_F')

    # table required: crosstab / iv ranking / individual
    if args.crosstab is not None:
        iscategorical = args.crosstab in bprofile.get_categorical_column()
        # replace target variable name with var name
        df = bprofile.get_crosstab(args.crosstab, categorical=iscategorical)
        df.index.name = args.crosstab
        df.to_csv(sys.stdout)

    else:
        # customize output
        if args.column is None:
            result = bprofile.iv_ranking()
            # check iv
            threshold = 0.15
            approved = "YES" if result.iv[0] > threshold else "NO"
            print(result)
            print("Are the two groups differ (heterogeneous)? {}".format(approved))
            print("NOTE: The two groups are heterogeneous if there exists a variable with an IV greater than {}".format(threshold))
        else:
            # determine if it is categorical
            iscategorical = args.column in bprofile.get_categorical_column()
            woe = bprofile.get_woe(args.column, categorical=iscategorical)
            # replace target variable name with var name
            woe.index.name = args.column
            woe.to_csv(sys.stdout)
            # iv = bprofile.get_iv(args.column, categorical=iscategorical)
            # print('<{}>: {:.{prec}%}'.format(args.column, iv, prec=4))
