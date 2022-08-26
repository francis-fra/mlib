import numpy as np

# global
LGD = 0.5

def get_demand_cols():
    "all columns required for demand model"
    demand_cols = ['Conversion_Status', 'Price_Rate', 'Log_Vol', 'Log_Assets', 'Log_Income', 'High_Activity', 'Male', 
            'Has_Assets', 'Had_Prior_Loans', 'Has_Bank_Credit_Card', 'score_group', 'Tenure', 'Age',
        'Has_CA', 'Has_Car', 'Income', 'Is_Employed', 'No_of_Activities_Last_1Year', 'Requested_Volume']
    return demand_cols

def get_pdiscount_cols():
    "all columns required for prob discount model"
    pdiscount_cols = ['No_of_Activities_Last_1Year', 'Price_Rate', 'Log_Vol', 'High_Activity', 'score_group']
    return pdiscount_cols

def get_discount_amt_cols():
    "all columns required for discount amt model"
    discount_amt_cols = ['Price_Rate', 'Has_Assets', 'PD', 'Log_Vol', ]
    return discount_amt_cols

def get_demand_data(df):
    "return dataframe required for demand model"
    sdf = df.copy()
    sdf['Male'] = sdf['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    sdf['High_Activity'] = sdf['Activity'].apply(lambda x: 1 if x == 'High' else 0)
    sdf['Log_Vol'] = np.log(sdf['Requested_Volume'] + 1)
    sdf['Log_Assets'] = np.log(sdf['Assets_Amount'] + 1)
    sdf['Log_Income'] = np.log(sdf['Income'] + 1)
    cols = ['Conversion_Status', 'Price_Rate', 'Log_Vol', 'Log_Assets', 'Log_Income', 'High_Activity', 'Male', 
        'Has_Assets', 'Had_Prior_Loans', 'Has_Bank_Credit_Card', 'score_group', 'Tenure', 'Age',
       'Has_CA', 'Has_Car', 'Income', 'Is_Employed', 'No_of_Activities_Last_1Year', 'Requested_Volume']
    return sdf[cols]

def get_pdiscount_data(df):
    "return dataframe required for pdiscount model"
    sdf = df.copy()
    sdf['High_Activity'] = sdf['Activity'].apply(lambda x: 1 if x == 'High' else 0)
    sdf['Log_Vol'] = np.log(sdf['Requested_Volume'] + 1)
    return sdf

def get_discount_amount_data(df):
    "return dataframe required for discount amt model"
    sdf = df.copy()
    sdf['Log_Vol'] = np.log(sdf['Requested_Volume'] + 1)
    return sdf

def get_kpi(df):
    "get portfolio statistics"
    # portfolio stat
    conversion_prob = df['demand'].mean()
    # volume = NAF
    avg_req_vol = df['Requested_Volume'].mean()
    total_req_vol = df['Requested_Volume'].sum()
    # replace conversion status with modeled conversion rate
    # total_booked_vol = (df['Requested_Volume'] * df['Conversion_Status']).sum()
    # avg_booked_vol = (df['Requested_Volume'] * df['Conversion_Status']).sum() / df['Conversion_Status'].sum()
    # portfolio volume
    total_booked_vol = (df['Requested_Volume'] * df['demand']).sum()
    # avg vol if converted
    avg_booked_vol = (df['Requested_Volume'] * df['demand']).sum() / df['demand'].sum()
    # income if all converted
    offered_income = (df['Requested_Volume']  * df['margin_rate']).sum()
    # portfolio income (profit)
    total_booked_income = (df['Requested_Volume']  * df['demand'] * df['margin_rate']).sum()
    vw_ratio = total_booked_income / total_booked_vol
    
    print('conversion prob : {:.2f}%'.format(conversion_prob*100))
    print('Margin Rate vol avg : {:.2f}%'.format(vw_ratio*100))
    print('Margin')    
    print('Margin income Booked Total : {:,.0f}'.format(total_booked_income))
    print('Margin income Offered Total : {:,.0f}'.format(offered_income))
    print('Volume')
    print('avg offered vol : {:,.0f}'.format(avg_req_vol))
    print('total offered vol : {:,.0f}'.format(total_req_vol))
    print('avg booked vol : {:,.0f}'.format(avg_booked_vol))
    print('total booked vol : {:,.0f}'.format(total_booked_vol))

def cal_transfer_rate(df):
    "transfer rate lookup table"
    is_score_less_than_9 = df['score_group'] < 9
    is_vol_less_than_15000 = df['Requested_Volume'] < 15000
    tt = np.logical_and(is_score_less_than_9, is_vol_less_than_15000)
    tf = np.logical_and(is_score_less_than_9, np.logical_not(is_vol_less_than_15000))
    ft = np.logical_and(np.logical_not(is_score_less_than_9), is_vol_less_than_15000)
    ff = np.logical_and(np.logical_not(is_score_less_than_9), np.logical_not(is_vol_less_than_15000))
    return (tt, tf, ft, ff)

def get_transfer_rate(df):
    "add transfer_rate columun"
    pricedf = df.copy()
    (tt, tf, ft, ff) = cal_transfer_rate(pricedf)
    pricedf['transfer_rate'] = 0
    pricedf.loc[tt, 'transfer_rate'] += 1.5/100
    pricedf.loc[ft, 'transfer_rate'] += 1.2/100
    pricedf.loc[tf, 'transfer_rate'] += 1.7/100
    pricedf.loc[ff, 'transfer_rate'] += 1.4/100
    return pricedf

def get_demand_from_model(df,demand_model):
    "add demand column"
    sdf = get_demand_data(df)
    demand_cols = get_demand_cols()
    sdf = sdf[demand_cols]
    pricedf = df.copy()
    pricedf['demand'] = demand_model.predict(sdf)
    return pricedf

def get_discount_amt_from_model(df, discount_model):
    "add pred_Discount_amt column"
    sdf = get_discount_amount_data(df)
    discount_amt_cols = get_discount_amt_cols()
    pricedf = df.copy()
    pricedf['pred_Discount_amt'] = discount_model.predict(sdf[discount_amt_cols])
    pricedf['pred_Discount_amt'] = pricedf['pred_Discount_amt'].fillna(0)
    return pricedf

def get_prob_discount_from_model(df, pdiscount_model):
    "add prob_discount column"
    sdf = get_pdiscount_data(df)
    pdiscount_cols = get_pdiscount_cols()
    pricedf = df.copy()
    pricedf['prob_discount'] =  pdiscount_model.predict(sdf[pdiscount_cols])
    return pricedf

def determine_price(df, models):
    """
        By using the demand, prob discount and discount amount models, 
        it calculates:
        costs:
            1) cost risk (pd x lgd)
            2) transfer rate
            3) expected discount 
        demand:
            4) conversion rate (demand)
        the margin rate for each customer  = price - costs
    """
    demand_model = models['demand_model']
    pdiscount_model = models['pdiscount_model']
    discount_model = models['discount_model']

    pricedf = df.copy()
    pricedf['cost_risk'] = pricedf['PD'] * LGD
    pricedf = get_transfer_rate(pricedf)
    pricedf = get_demand_from_model(pricedf, demand_model)
    pricedf = get_discount_amt_from_model(pricedf, discount_model)
    pricedf = get_prob_discount_from_model(pricedf, pdiscount_model)

    pricedf['Expected_Discount'] = pricedf['prob_discount'] * pricedf['pred_Discount_amt']

    # int rate param is set as the price rate in data
    pricedf['margin_rate'] = pricedf['Price_Rate'] - pricedf['transfer_rate'] - pricedf['Expected_Discount'] - pricedf['cost_risk']
    pricedf['margin_rate'] = pricedf['margin_rate'].fillna(0)
    return pricedf