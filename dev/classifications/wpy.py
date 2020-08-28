import pandas as pd
import numpy as np
import pyodbc, os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def connect_td(password, userid=os.environ["USERNAME"]):
    "get TD connection cursor"

    connect_string = 'DSN=tdp5;UID=' + userid + ';PWD=' + password
    cnxn = pyodbc.connect(connect_string)
    cnxn.autocommit = True
    return cnxn

def col_upper_case(df):
    data_cols = df.columns.values.tolist()
    df.columns = [col.upper() for col in data_cols]

    return df

def get_single_column(df, col):
    return np.ravel(df[col])

class VarDropper(TransformerMixin):

    def __init__(self, excl=['TARGET_F', 'CUSTOMER_ID', 'REF_MONTH', 'DATA_DT', 'PROCESS_DTTM', 'ACCT_KEY', 'ACCOUNT_ID', 'GCIS_KEY']):
        self._excl = excl

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # cols_to_drop = list(intersect(set(X.columns), set(self._excl)))
        cols_to_drop = list((set(X.columns).intersection(self._excl)))
        return X.drop(cols_to_drop, axis=1, inplace=False)


class Imputer(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        self.categoricalCols = [col for col in X.dtypes.index if X.dtypes.get(col) == 'object' ]

        do_transform = lambda x: x.fillna('NA') \
                if x.name in self.categoricalCols else x.fillna(0)

        result = X.copy()
        result = result.apply(do_transform)

        return result

class Encoder(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        self.categoricalCols = [col for col in X.dtypes.index if X.dtypes.get(col) == 'object' ]
        self.fit_dict = defaultdict(LabelEncoder)

        # lambda function either transform or a columns or return the same column
        do_transform = lambda x: self.fit_dict[x.name].fit_transform(x) \
                    if x.name in self.categoricalCols else x

        result = X.copy()

        # Encoding and return results
        result = result.apply(do_transform)

        return result


def print_model_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)

    print("Accuracy: %f" % accuracy_score(y_test, predictions))
    print("Confustion Matrix: ")
    print(confusion_matrix(y_true=y_test, y_pred=predictions))
    print("AUC Score: %f" % roc_auc_score(y_test, pred_prob[:,1]))
    print("GINI Coefficient: %f" % (roc_auc_score(y_test, pred_prob[:,1])*2-1))


def plot_roc(model, model_label, X_test, y_test, save_img=False, img_path=''):
    pred_prob = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, pred_prob[:,1])
    gini = 2*roc_auc - 1
    fpr, tpr, thresholds = roc_curve(y_test, pred_prob[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label = model_label + ' (AUC = {0:4.2}, GINI = {1:4.2})'.format(roc_auc, gini))

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.gca().set_facecolor((0.92, 0.92, 0.92))
    plt.gcf().set_facecolor((0.82, 0.82, 0.82))
    plt.gcf().set_size_inches(7, 5)

    if (save_img == True) :
        plt.savefig(img_path)
    plt.show()

def get_var_importance(model, features, top):
    "Get sorted top var importance"

    d = {'importance': model.feature_importances_}
    varimp = pd.DataFrame(d, index=features)
    varimp = varimp.sort_values('importance', ascending=False).head(top)
    return varimp

def write_scoring_sql(top_features, td_score_population_name, td_score_table_name, score_sql_file_name):

    sql_string = """

    CREATE TABLE {{SCORE_TABLE_NAME}} AS
    (

      SELECT #A.*
        {{KEY_FEATURES}}
      FROM
      (
      	SELECT
          GCIS_KEY
      		,data_dt
      	FROM {{SCORE_POPULATION_TABLE_NAME}}
      	WHERE GCIS_KEY IS NOT NULL
      ) as #A
      JOIN C4P1VPAF.MPA_SGB_MDM_GCIS_BASE as #B
      ON #A.GCIS_KEY=#B.GCIS_KEY
      AND #A.DATA_DT=#B.DATA_DT

    ) WITH DATA
    UNIQUE PRIMARY INDEX (GCIS_KEY) ;

    """

    feature_str = ""
    for name in top_features.index:
        feature_str += ", " + name + "\n"

    sql_string = sql_string.replace("{{SCORE_TABLE_NAME}}", td_score_table_name)
    sql_string = sql_string.replace("{{SCORE_POPULATION_TABLE_NAME}}", td_score_population_name)
    sql_string = sql_string.replace("{{KEY_FEATURES}}", feature_str)

    with open(score_sql_file_name, "w") as f:
        f.write(sql_string)

# adopted from Joe for standardized serialization
class ModelWrapper(object):

    def __init__(self, model, features, pipeline, prob_train=0.5, prob_prior=0.5, algorithm='',
                 model_name='', cust_key=['CUSTOMER_ID'], version='1.0', build_month='ccyymm', built_by=''):
        self.model = model
        self.features = features
        self.pipeline = pipeline
        self.prob_train = prob_train
        self.prob_prior = prob_prior
        self.algorithm = algorithm
        self.model_name = model_name.upper()
        self.cust_key = [k.upper() for k in cust_key]
        self.version = version
        self.build_month = build_month
        self.built_by = built_by
        self.original_feature = self.get_original_feature(features)

    def get_original_feature(self, features):
        features_orginal = []

        for s in features:
            i = s.find("#")
            if (i == -1): features_orginal.append(s)
            else:
                v = s[0 : i]
                if (v not in features_orginal): features_orginal.append(v)

        return features_orginal

    def predict_proba(self, X):
        X.columns = [col.upper() for col in X.columns.values.tolist()]
        #score = X[[self.cust_key.upper()]]
        score = X[self.cust_key]
        X_pop = self.pipeline.transform(X)
        pred_prob = self.model.predict_proba(X_pop)

        for r in range(len(pred_prob)):
            pred_prob[r][1] = pred_prob[r][1] * (self.prob_prior / self.prob_train)
            pred_prob[r][0] = pred_prob[r][0] * ((1-self.prob_prior) / (1-self.prob_train))
            t = pred_prob[r][1] + pred_prob[r][0]
            pred_prob[r][1] = pred_prob[r][1] / t
            pred_prob[r][0] = pred_prob[r][0] / t

        score['SCORE'] = pred_prob[:,1]
        score['SCORE'] = score['SCORE'].apply(lambda x: round(x, 6))
        score['PCTILE'] = score['SCORE'].rank(ascending=False, pct=True, method='min')
        score['PCTILE'] = np.ceil(score['PCTILE']*100).astype(int)

        return score

def get_decision(probas, score_cutoff, percentile_cutoff):
    "hard decision for each customer"

    score = probas[:,1]

    # decision based on probability
    if (score_cutoff is None and percentile_cutoff is None):
        decision = score > probas[:,0]
    elif (score_cutoff is not None):
        decision = score >= score_cutoff
    else:
        # decision based on percentile cutoff otherwise
        num_bins = 100
        ranking =  pd.qcut(-probas[:,1], 100 , labels=np.linspace(1, 100, num=100))
        decision = [1 if rank <= percentile_cutoff else 0 for rank in ranking]

    return decision

def get_score_table_columns(brand):
    "get scoring table column names"

    if brand == 'WBC':
        col_name = ['DATA_DT', 'CUSTOMER_ID', 'ACCOUNT_ID', 'MODEL_NAME', 'SCORE', 'PCTILE', 'PREDICT', 'PROCESSED_DTTM']
    else:
        col_name = ['DATA_DT', 'GCIS_KEY', 'ACCT_KEY', 'MODEL_NAME', 'SCORE', 'PCTILE', 'PREDICT', 'PROCESSED_DTTM']
    return col_name

def create_scoring_data_frame(brand, model_id, cust_id, data_dt, probas, percentile_cutoff=20, score_cutoff=None, acct_id=None):
    "create data frame matching with the scoring table"

    score = probas[:,1]

    #
    data_dt_str = data_dt.strftime("%Y-%m-%d")
    # 100 for percentile
    num_bins = 100
    pctile = pd.qcut(-score, num_bins, labels=np.linspace(1, num_bins, num=num_bins))
    # data time stamp
    processed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # decision
    decision = get_decision(probas, score_cutoff, percentile_cutoff)
    # scoring table column name
    col_name = get_score_table_columns(brand)

    score_data = {
                    col_name[0] : data_dt_str,
                    col_name[1] : cust_id,
                    col_name[2] : acct_id,
                    col_name[3] : model_id,
                    col_name[4] : score,
                    col_name[5] : pctile,
                    col_name[6] : decision,
                    col_name[7] : processed_date
                }

    df = pd.DataFrame.from_dict(score_data)
    return df[col_name]
