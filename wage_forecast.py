#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:52:30 2019

@author: c0s02bi
"""

#!/usr/bin/env python
# coding: utf-8
# Author: Chao Song 2019/01/08


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from fbprophet import Prophet
import math
from pydlm import dlm, trend, seasonality
from pydlm import dynamic,autoReg,modelTuner
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats.stats import pearsonr  
from fbprophet.plot import plot_yearly
from datetime import timedelta



calendar= pd.read_csv('wm_yr_wk_ref.csv',parse_dates=[0], names=['calendar_date','wm_yr_wk_nbr'],header=0)

def removehurricane(change_col,dfc,start_date,end_date,sales = False):
    '''
      for wage data, change_col = 'cost', sales = flase
      for sales data, change_col = 'total_sales',sales = True
      the window size = end_date & start_date is given by Didun's insight, manually???
      Reference: https://confluence.walmart.com/display/SASTDSE/Hurricane+adjustment
    '''
    date_col = 0
    club_col = 0
    df = dfc.copy(deep=True)
    target_col = df.columns.tolist().index(change_col)
    if sales:
        date_col = df.columns.tolist().index('WM_YEAR_WK_NBR')
        club_col = df.columns.tolist().index('club')
    else:
        date_col = df.columns.tolist().index('wm_yr_wk_nbr')
        df.columns.tolist().index('club_nbr')
    club_ls = df[df.columns[club_col]].unique().tolist()
    coeff = []  # the inflation rate of sales nearby the period
    for club in club_ls:
        coeff.append(1)  # set some default number for safety
        start_date_low = wm_nbr_add(start_date,-6)
        end_date_high = wm_nbr_add(end_date,6)
        start_date_last = wm_nbr_add(start_date,-52)
        end_date_last = wm_nbr_add(end_date,-52)
        start_date_low_last = wm_nbr_add(start_date_low,-52)
        end_date_high_last = wm_nbr_add(end_date_high,-52)
        coeff[-1] = df.loc[(df[df.columns[club_col]]==club)&(((df[df.columns[date_col]]>=start_date_low)&(df[df.columns[date_col]]<start_date)) | ((df[df.columns[date_col]]>end_date)&(df[df.columns[date_col]]<=end_date_high)))][change_col].mean()
        coeff[-1] = df.loc[(df[df.columns[club_col]]==club)&(((df[df.columns[date_col]]>=start_date_low_last)&(df[df.columns[date_col]]<start_date_last)) | ((df[df.columns[date_col]]>end_date_last)&(df[df.columns[date_col]]<=end_date_high_last)))][change_col].mean()/coeff[-1]
        # Hurricane effect date is recovered by inflation_rate (=coeff[club]) * LY_values
    for j in range(df.shape[0]):
        if df.iloc[j,date_col] >= start_date and df.iloc[j,date_col] <= end_date:
            tmp = df.loc[(df[df.columns[club_col]]==df.iloc[j,club_col]) & (df[df.columns[date_col]]==wm_nbr_add(df.iloc[j,date_col],-52))]
            club = df.iloc[j,club_col]
            df.iloc[j,target_col] = coeff[club_ls.index(club)]*tmp.iloc[0,target_col]          
    
    return df
    

def getDatesFromWMWks(wm_wk_list):
    ''' given a list of WM weeks, e.g. 201505, 201631, etc., returns dates
        for the Fridays in the weeks
        From Michael\'s code'''
    ref_date = pd.datetime(2018,2,2)  # corresponding to 201801
    ref_wk   = 201801
    def weekDiff(target, ref):
        # both target and reference in YYYYWW format e.g. 201840 for 40th week of 2018
        target_str = str(target); ref_str = str(ref)
        target_yr = int(target_str[:4]); ref_yr = int(ref_str[:4])
        target_wk = int(target_str[4:]); ref_wk = int(ref_str[4:])
        assert (len(target_str) == 6) and (len(ref_str) == 6) and (target_wk < 53) and (ref_wk < 53)
        wk_diff = (target_yr-ref_yr) * 52 + (target_wk-ref_wk)
        return wk_diff
    dates = [ref_date + pd.Timedelta('7 days') * weekDiff(x, ref_wk) for x in wm_wk_list]
    return dates

def wm_nbr_add (cur,weeks,cal = calendar):
    '''
    addition for wm_year_wk_nbr
    '''
    cur = [cur]
    cur_date = getDatesFromWMWks(cur)
    fu_date = cur_date[0] + timedelta(days=7*weeks)
    fu_wk = cal.loc[cal.calendar_date == fu_date].iloc[0,1]
    
    return fu_wk


def prep_data(raw,storemap,scat_op_club):
    '''
    Seperate the data by their categries.
    From Siddarth's code
    
    '''
    #raw= pd.read_csv('./data/sap_data_20181213.csv')
    valid_ref_doc_nums = [ 50, 2115, 8, 51, 2307, 3438, 2705, 263432, 3227, 82, 2901, 3413, 54, 3238]
    ref_doc_dict = { 50:'Bereavement', 2115:'Disaster', 8:'Holiday Bonus', 51:'Jury Duty',
               2307:'Lump Sum Hourly', 3438:'Non-worked Hours', 2705:'OT pay Hist Award',
               263432:'Overtime', 3227:'Personal Payout Unused', 82:'Personal Time',
               2901:'Regional Pay Zone', 3413:'Retro Reg Hours', 54:'Severance Pay',
               3238:'Term Net Overpayment'}

    df= raw[raw.iloc[:,2].isin([3413, 2307, 8, 54, 291517,263432,50,51,82])].copy()
    cols= df.columns
    df=df.drop(cols[:2],axis=1)
    df= df.drop(cols[3:5],axis=1)
    df.columns= ['rf_doc_num','posting_date','cost_center','cost','retail']
    df['club_nbr']= df.iloc[:,-3].str.extract('[US|PR]0(\d+).*G.*$')
    df= df.loc[~df.club_nbr.isnull()]
    df.club_nbr= df.club_nbr.astype(int)
    df['country']= df.cost_center.str.extract('CONA/(.{2}).*$')
    df.posting_date=pd.to_datetime(df.posting_date)
    df=df.drop('cost_center',axis=1)
    df_PR = df[df.country.isin(['PR'])]
    df_PR = df_PR.sort_values(by=['club_nbr','posting_date'])    
    df = df_PR
    storemap = storemap.loc[storemap['state_prov_code']=='PR']
    df=pd.merge(left=df, right= storemap, how='left', left_on= 'club_nbr', right_on='club_nbr', validate= 'many_to_one')
    #raw= pd.read_csv('./data/sap_data_20181213.csv')
    #storemap= pd.read_csv('./storemap.csv', usecols=[1,2,3,4,5])
    #scat_op_club= pd.read_csv('./scat_op_club.csv', usecols=[2] )
    df=df.merge(right= scat_op_club, how='inner', left_on= 'club_nbr', right_on= 'CLUB_NBR')
    df=df.drop('CLUB_NBR', axis=1)
    #Subsetting for valid data
    df= df[df.open_status_code!=7]
    df= df[~df.open_status_code.isna()].drop('open_status_code',axis=1)
    df= df[df.country=='PR']
    calendar= pd.read_csv('wm_yr_wk_ref.csv',parse_dates=[0], names=['calendar_date','wm_yr_wk_nbr'],header=0)
    df=pd.merge(left=df, right= calendar, how='left', left_on='posting_date', right_on= 'calendar_date').drop('calendar_date',axis=1)
    df['date_pd']= getDatesFromWMWks(df.wm_yr_wk_nbr)
    punched_df= df[(df.rf_doc_num==291517) | (df.rf_doc_num== 263432)].copy()
    residual_worked_df = df[(df.rf_doc_num==50) | (df.rf_doc_num== 51) | (df.rf_doc_num==82)].copy()
    retro_df= df[df.rf_doc_num==3413].copy().drop(['rf_doc_num'],axis=1)
    holiday_df= df[df.rf_doc_num==8].copy().drop(['rf_doc_num'],axis=1)
    lump_df= df[df.rf_doc_num==2307].copy().drop(['rf_doc_num'],axis=1)
    severance_df= df[df.rf_doc_num==54].copy().drop(['rf_doc_num'],axis=1)
    punched_df = punched_df.sort_values(by='posting_date')
    return [punched_df,residual_worked_df,retro_df,holiday_df,lump_df,severance_df,calendar]

# # Modeling the Punched Data



# this part is left for Prophet model without adding any regressor
# two options using bi-weekly view or using daily view
day_sep = [0.1464,0.1480,0.1460,0.1067,0.1463,0.1489,0.1577,0.1464,0.1480,0.1460,0.1067,0.1463,0.1489,0.1577]# Sat,..., Thur, Fri
def gen_daily_data(punched_pro_club,day_sep):
    '''  
     to generate the daily data, since the distribution across weekday is quite consistent.
     day_sep obtained by US_clubs
     daily view data can also be feed into US_clubs_estimator?
    '''
    dt_ls = []
    ct_ls = []
    wk_ls = []
    for j in range(punched_pro_club.shape[0]):
        cur_date = punched_pro_club.iloc[j,0]-timedelta(days = 13) # the first payperiod
        for i in range(14):
            if (dt_ls.count(cur_date) > 0):
                print("shoot")
            dt_ls.append(cur_date)
            cur_date = cur_date+timedelta(days = 1)
            ct_ls.append(punched_pro_club.iloc[j,1]*day_sep[i]/2)
            wk_ls.append(punched_pro_club.iloc[j,2])
    res = pd.DataFrame({'posting_date':dt_ls,'cost':ct_ls,'wm_yr_wk_nbr':wk_ls})
    return res
def estimate_and_predict_prophet_PR(calendar,punched_df, end_train_date, start_test_date, daily_view=False, target_column = 'cost',pred_days=120,horizon = 8,missing_val = 201735):
    ''' 
        Using facbook prophet model without any regressor
        'daily_view' variable is an indicator specified by user whether to seperate bi-weekly SAP data to daily
        'daily_view' is not recommended.
        'pred_days' variable is how many days ahead you want to predict
        return type: prediction result as a DataFrame, 
        columns=['ds','yhat','club'] ds is the posting_date and yhat is the prediction value
        this serves as the first layer of mixed model.
    '''
    if 'club_nbr' not in punched_df.columns:
        punched_df['club_nbr'] = punched_df['club']
        punched_df = punched_df.drop('club',axis = 1)
    if 'posting_date' not in punched_df.columns:
        punched_df['posting_date'] = getDatesFromWMWks(punched_df['wm_yr_wk_nbr'])    
    punched = punched_df.groupby(['club_nbr','posting_date'])[target_column].sum()
    punched.column = ['total_punched_wg']
    punched = punched.reset_index()
    punched = pd.merge(left=punched, right=calendar, how='left', left_on='posting_date', right_on= 'calendar_date').drop('calendar_date',axis=1)
    punched = punched.drop('posting_date',axis = 1)
    punched_pro = punched_df.groupby(['club_nbr','posting_date'])[target_column].sum()
    punched_pro.column = ['total_punched_wg']
    punched_pro = punched_pro.reset_index()
    punched_pro = pd.merge(left=punched_pro, right=calendar, how='left', left_on='posting_date', right_on='calendar_date').drop('calendar_date',axis=1)
    punched_pro = removehurricane(target_column,punched_pro,201733,201739,sales = False)  #201735 is missing in the SAP data, recover below     
    club_ls = punched_pro.club_nbr.unique()
    res = pd.DataFrame()
    for club in club_ls:
        cur = club
        punched_pro_club = punched_pro[punched_pro.club_nbr.isin([club])]
        ##########################################
        #adding missing value
        if missing_val not in punched_pro_club.wm_yr_wk_nbr.values.tolist():
            punched_pro_club.loc[-1] = [club,punched_pro_club.loc[punched_pro_club.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,1]+timedelta(days=14),0.5*punched_pro_club.loc[punched_pro_club.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,2]+0.5*punched_pro_club.loc[punched_pro_club.wm_yr_wk_nbr==wm_nbr_add(missing_val,2)].iloc[0,2],missing_val]  # adding a row
            punched_pro_club.index = punched_pro_club.index + 1  # shifting index
        ##############################################
        punched_pro_club = punched_pro_club.sort_values(by='wm_yr_wk_nbr')
        punched_pro_club = punched_pro_club.drop('club_nbr',axis=1).reset_index().drop('index',axis=1)
        if (daily_view):
            punched_pro_club = gen_daily_data(punched_pro_club,day_sep)
        trainset = punched_pro_club.loc[punched_pro_club.wm_yr_wk_nbr<=end_train_date].drop(['wm_yr_wk_nbr'],axis=1)
        columnsTitles=["posting_date",target_column]
        trainset=trainset.reindex(columns=columnsTitles)
        trainset.columns=["ds","y"]
        m = Prophet(yearly_seasonality=True)
        
        m.fit(trainset)
        future = m.make_future_dataframe(periods=pred_days)
        forecast = m.predict(future)
        result = forecast[['ds','yhat']].tail(pred_days)
        weeklist = []
        for i in range(horizon):
            weeklist.append(trainset.iloc[-1,trainset.columns.tolist().index('ds')] + timedelta(days=14*(i+1)))
        result = result[result.ds.isin(weeklist)]
        yhat = result.yhat.values
        if res.shape[0] == 0:
            tmp = result
            tmp['club'] = pd.Series([cur for i in range(result.shape[0])],index=tmp.index)
            res = tmp
        else:
            tmp = result
            tmp['club'] = pd.Series([cur for i in range(result.shape[0])],index=tmp.index)
            res = pd.concat([res,tmp],axis = 0)
    return res


def proportion(df_sales):
    '''
      this is used as seperation of each club's sales among a subset of clubs together (PR clubs here)
      return type: Dataframe with proportion 
      return columns list = [total_sales_across  Unnamed: 0  club   per_nbr  wm_yr_wk_nbr]
      total_sales_across is the # in the Puerto Rico region on that day.
      per_nbr is the proportion of this club accounted for among total_sales_across
    '''
    ######################
    '''
    Hurricane Adjusemnt period: 09-15-2017 to 10-20-2017
    
    
    '''
    club_ls = df_sales.club.unique()
    df_sales = removehurricane('total_sales',df_sales,201733,201739,sales = True)
    df_total_sales = df_sales.groupby('WM_YEAR_WK_NBR')['total_sales'].sum()
    df_total_sales = df_total_sales.reset_index()
    df_total_sales.columns = ['WM_YEAR_WK_NBR','total_sales_across']
    df_propor = pd.merge(left=df_total_sales,right=df_sales,left_on='WM_YEAR_WK_NBR',right_on='WM_YEAR_WK_NBR',validate ='1:m')
    df_propor['per_nbr'] = df_propor['total_sales']/df_propor['total_sales_across']
    df_propor = df_propor.drop('total_sales',axis = 1)
    df_propor = df_propor.sort_values(by = ['club','WM_YEAR_WK_NBR']).reset_index().drop('index',axis = 1)
    df_propor['wm_yr_wk_nbr'] = df_propor['WM_YEAR_WK_NBR']
    df_propor = df_propor.drop(['WM_YEAR_WK_NBR'],axis = 1)
    return df_propor

def predict_proportion(calendar,df_sales,end_train_date, start_test_date,horizon = 8):
    '''
      treat proportion as a time series and use as obseravle regressor in dlm data
      
      return: dataframe contains the future dates combined with actual data. 
      
      return dataframe columns_list = [club,wm_yr_wk_nbr,per_nbr_fc,total_sales_across]
      
      date before start_test_date is true value, althought under column 'per_nbr_fc'.
      
      date after start_test_date is prediction value. It is obtained at once before we go to dlm model for the wage prediction. 
      
    '''
    df_propor = proportion(df_sales)
    df_propor_PR_ts = pd.DataFrame()
    club_ls = df_propor.club.unique()
    for club in club_ls: 
        df_propor_club = df_propor[df_propor.club.isin([club])]
        trainset_propor = df_propor_club.loc[df_propor_club.wm_yr_wk_nbr<=end_train_date]
        predictMean = estimate_and_predict_prophet_PR(calendar,df_propor_club, end_train_date, start_test_date, target_column = 'per_nbr')
        predictMean = pd.merge(left=predictMean,right=calendar,left_on='ds',right_on='calendar_date',validate='1:1').drop(['ds','calendar_date'],axis = 1)
        totalMean = estimate_and_predict_prophet_PR(calendar,df_propor_club, end_train_date, start_test_date, target_column = 'total_sales_across')
        totalMean = pd.merge(left=totalMean,right=calendar,left_on='ds',right_on='calendar_date',validate='1:1').drop(['ds','calendar_date'],axis = 1)
        l = trainset_propor.shape[0]+horizon
        wk_ls = trainset_propor.wm_yr_wk_nbr.values.tolist()
        wk_ls = wk_ls + [wm_nbr_add(start_test_date,x) for x in range(0,horizon*2,2)]
        trainset_propor = df_propor_club.loc[df_propor_club.wm_yr_wk_nbr<=end_train_date]
        tmp = pd.DataFrame({'club':[club for i in range(l)],'wm_yr_wk_nbr':wk_ls,'per_nbr_fc':trainset_propor['per_nbr'].values.tolist()+predictMean['yhat'].values.tolist()})
        tmp['total_sales_across'] = pd.Series(trainset_propor['total_sales_across'].values.tolist()+totalMean['yhat'].values.tolist())
        if (df_propor_PR_ts.shape[0] == 0):
            df_propor_PR_ts = tmp.copy(deep = True)
        else:
            df_propor_PR_ts = pd.concat([df_propor_PR_ts,tmp],axis = 0) 
    df_propor_PR_ts = df_propor_PR_ts.reset_index().drop('index',axis = 1)
    return df_propor_PR_ts


# using proportion of sales as the forecast for the real sales (more predictable)
# the total trend is included in macro variable (assumption)
def estimate_and_predict_dlm_PR(calendar,df_propor_PR_ts, punched_df, end_train_date, start_test_date,start_of_this_year,enable_sales,pred_weeks = 8,locality = 10,r = 0.05,missing_val = 201735):
    '''
       accept the forecasting sales_proportion data as one regressor
       df_propor_PR_test: []
       return type: DataFrame with prediction result
       return: columns = [wm_yr_wk_nbr,club,yhat]
    
    '''
    res = pd.DataFrame()
    punched = punched_df.groupby(['club_nbr','posting_date'])['cost'].sum()
    punched.column = ['total_punched_wg']
    punched = punched.reset_index()
    punched = pd.merge(left=punched, right=calendar, how='left', left_on='posting_date', right_on= 'calendar_date').drop('calendar_date',axis=1)
    # mean wage among all clubs
    punched = removehurricane('cost',punched,201733,201739,sales = False)  
    punched_mean = punched.groupby(['wm_yr_wk_nbr','posting_date'])['cost'].mean()
    punched_mean = punched_mean.reset_index()
    punched_mean.columns = ['wm_yr_wk_nbr','posting_date','cost']
    punched_mean['club_nbr'] = pd.Series(np.ones([punched_mean.shape[0]]))
    ##########################
    if missing_val not in punched_mean.wm_yr_wk_nbr.tolist():
        punched_mean.loc[-1] = [missing_val,punched_mean.loc[punched_mean.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,1]+timedelta(days=14),0.5*punched_mean.loc[punched_mean.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,2]+0.5*punched_mean.loc[punched_mean.wm_yr_wk_nbr==wm_nbr_add(missing_val,2)].iloc[0,2],1]  # adding a row
        punched_mean.index = punched_mean.index + 1
    #########################
    punched_mean1 = punched_mean.copy(deep=True)
    punched_mean1['cost'] = 0.5*punched_mean1['cost']+ 0.25*punched_mean1['cost'].shift(1)+0.25*punched_mean1['cost'].shift(2)
    ty = punched_mean1['cost'].mean()
    punched_mean1[['cost']] = punched_mean1[['cost']].fillna(value = ty)
    punched_mean1 = estimate_and_predict_prophet_PR(calendar,punched_mean1, end_train_date, start_test_date, daily_view=False, pred_days=120) #predict the mean wages.
    punched_mean1 = punched_mean1.drop('club',axis = 1)
    punched_mean1.columns = ['posting_date','PR_cost']
    punched_mean1 = pd.merge(left=punched_mean1,right=calendar,how = 'left',left_on='posting_date',right_on='calendar_date').drop('calendar_date',axis=1)
    tmp = punched.groupby(['wm_yr_wk_nbr','posting_date'])['cost'].mean()
    tmp = tmp.reset_index()
    tmp.columns = ['wm_yr_wk_nbr','posting_date','PR_cost']
    tmp = tmp.loc[tmp.wm_yr_wk_nbr<=end_train_date]
    tmp['PR_cost'] = 0.5*tmp['PR_cost']+0.25*tmp['PR_cost'].shift(1)+0.25*tmp['PR_cost'].shift(2)
    ty = tmp['PR_cost'].mean()
    tmp[['PR_cost']] = tmp[['PR_cost']].fillna(value = ty)
    
    
    punched_mean = pd.concat([tmp,punched_mean1],axis = 0)
    if missing_val not in punched_mean.wm_yr_wk_nbr.tolist():
        tu = [0.5*punched_mean.loc[punched_mean.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,0]+0.5*punched_mean.loc[punched_mean.wm_yr_wk_nbr==wm_nbr_add(missing_val,2)].iloc[0,0]]
        tu.append(punched_mean.loc[punched_mean.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,1]+timedelta(days=14))
        tu.append(missing_val)
        punched_mean.loc[-1] = tu  # adding a row
        punched_mean.index = punched_mean.index + 1  # shifting index
    punched_mean = punched_mean.sort_values(by='wm_yr_wk_nbr').reset_index().drop('index',axis=1)
    punched = punched.drop('posting_date',axis = 1)
    punched_pro = punched_df.groupby(['club_nbr','posting_date'])['cost'].sum()
    punched_pro.column = ['total_punched_wg']
    punched_pro = punched_pro.reset_index()
    punched_pro=pd.merge(left=punched_pro, right= calendar, how='left', left_on='posting_date', right_on= 'calendar_date').drop('calendar_date',axis=1)
    punched_pro = removehurricane('cost',punched_pro,201733,201739,sales = False)    
            #201735 is Maria Hurrican Missing
            #201737 is the Irma Hurricane
    club_ls = punched.club_nbr.unique()
    for club in club_ls:
        pro_club = punched_pro[punched_pro.club_nbr.isin([club])]
        #########################################
        # adding missing value
        if missing_val not in pro_club.wm_yr_wk_nbr.tolist():
            pro_club.loc[-1] = [club,pro_club.loc[pro_club.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,1]+timedelta(days=14),0.5*pro_club.loc[pro_club.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,2]+0.5*pro_club.loc[pro_club.wm_yr_wk_nbr==wm_nbr_add(missing_val,2)].iloc[0,2],missing_val]  # adding a row
            pro_club.index = pro_club.index + 1  # shifting index
        ####################################################
        pro_club = pro_club.sort_values(by='posting_date').reset_index().drop('index',axis=1)
        pro_sales = df_propor_PR_ts.loc[df_propor_PR_ts.club == club].drop(['club'],axis=1)
        pro_club = pro_club.drop(['club_nbr','posting_date'],axis=1)
        pro_club.columns = ['cost','wm_yr_wk_nbr']
        pro_sales['total_sales'] = pro_sales['total_sales_across']*pro_sales['per_nbr_fc']
        pro_sales = pd.concat([pro_sales]+[pro_sales.total_sales.shift(x) for x in range(1,3)],axis=1)
        pro_sales.columns = ['wm_yr_wk_nbr','per_nbr_fc','total_sales_across','total_sales_0','sr_1','sr_2']
         #########################################
        # adding missing value
        if missing_val not in pro_sales.wm_yr_wk_nbr.unique().tolist():
            tu = []
            for k in range(len(pro_sales.columns)):
                tu.append(0.5*pro_sales.loc[pro_sales.wm_yr_wk_nbr==wm_nbr_add(missing_val,-2)].iloc[0,k]+0.5*pro_sales.loc[pro_sales.wm_yr_wk_nbr==wm_nbr_add(missing_val,2)].iloc[0,k])
            tu[0] = int(tu[0])
            pro_sales.loc[-1] = tu  # adding a row
            pro_sales.index = pro_sales.index + 1  # shifting index
        pro_sales = pro_sales.sort_values(by='wm_yr_wk_nbr').reset_index().drop('index',axis=1)
        pro_sales = pd.merge(left=pro_sales, right=punched_mean, how ='right',left_on='wm_yr_wk_nbr', right_on='wm_yr_wk_nbr', validate='1:1')
        pro_sales = pro_sales.drop(['posting_date'],axis=1)
        pro_sales = pro_sales.apply(lambda x: x.fillna(x.mean()),axis=0)
        pro_sales_train = pro_sales.loc[pro_sales.wm_yr_wk_nbr<=end_train_date]
        pro_sales_test = pro_sales.loc[pro_sales.wm_yr_wk_nbr>=start_test_date]
        # trend 
        linear_trend = trend(degree=2, discount=0.98, name='linear_trend', w=8)
        # seasonality
        seasonal26 = seasonality(period=26, discount=1, name='seasonal26', w=12)
        # control variable
        sales0 = pro_sales_train['total_sales_0'].values.tolist()
        s0 = [[x] for x in sales0]
        sales1 = pro_sales_train['sr_1'].values.tolist()
        s1 = [[x] for x in sales1]
        sales2 = pro_sales_train['sr_2'].values.tolist()
        s2 = [[x] for x in sales2]
        macro = pro_sales_train['PR_cost'].values.tolist()
        m1 = [[x] for x in macro]
        #####################################
        s0 = dynamic(features=s0, discount=0.99, name='sales0', w=8)
        s1 = dynamic(features=s1, discount=0.99, name='sales1', w=6) # use the actual sales and forecasting sales amount
        s2 = dynamic(features=s2, discount=0.95, name='sales2', w=6)
        m1 = dynamic(features=m1, discount=0.99, name='macro', w=12)
        
        #e1 = dynamic(features=e1,discount=0.95,name='eff',w=6)
        drm = dlm(pro_club['cost']) + linear_trend + seasonal26+autoReg(degree=locality, name='ar2', w=6)+m1#+s0+s1+s2+m1
        drm.fit()
        #testset
        pro_sales_test = pro_sales_test.head(pred_weeks)
        sales0test = pro_sales_test['total_sales_0'].head(pred_weeks).values.tolist()
        s0test = [[x] for x in sales0test]
        sales1test = pro_sales_test['sr_1'].head(pred_weeks).values.tolist()
        s1test = [[x] for x in sales1test]
        sales2test = pro_sales_test['sr_2'].head(pred_weeks).values.tolist()
        s2test = [[x] for x in sales2test]
        macrotest = pro_sales_test['PR_cost'].head(pred_weeks).values.tolist()
        m1test = [[x] for x in macrotest]
        #efftest = testset['eff'].head(pred_weeks).values.tolist()
        #e1test = [[x] for x in efftest]
        features = {'sales0':s0test,'sales1':s1test, 'sales2':s2test,'macro':m1test}#,'eff':e1test}
        (predictMean, predictVar) = drm.predictN(N=pred_weeks, date=drm.n-1,featureDict=features)
        #locality
        pro_sales = pro_sales.drop(['sr_1','sr_2'],axis = 1)
        pro_sales['ratio'] = pro_sales['total_sales_0']/pro_sales['total_sales_across']
        pro_sales['ratio_1'] = pro_sales['ratio'].shift(1)
        pro_sales['ratio_2'] = pro_sales['ratio'].shift(2)
        trainset1_year = pro_club.loc[pro_club.wm_yr_wk_nbr<=end_train_date].loc[pro_club.wm_yr_wk_nbr>=end_train_date-locality]
        trainset_year = pro_sales.loc[pro_sales.wm_yr_wk_nbr<=end_train_date].loc[pro_sales.wm_yr_wk_nbr>=end_train_date-locality]
        trainset_year.apply(lambda x: x.fillna(x.mean()),axis=0)
        linear_trend_year = trend(degree=1, discount=0.99, name='linear_trend_year', w=10)
        sales0_year = trainset_year['ratio'].values.tolist()
        s0_year = [[x] for x in sales0_year]
        # use the forecast of the ratio of each club among total in PR area
        # since this is a local model, the total amount in area can be assumed to be constant.
        sales1_year = trainset_year['ratio_1'].values.tolist() 
        s1_year = [[x] for x in sales1_year]
        sales2_year = trainset_year['ratio_2'].values.tolist()
        s2_year = [[x] for x in sales2_year]
        macro_year = trainset_year['PR_cost'].values.tolist()
        m1_year = [[x] for x in macro_year]
        #####################################
        s0_year = dynamic(features=s0_year, discount=0.99, name='sales0_year', w=10)
        s1_year = dynamic(features=s1_year, discount=0.99, name='sales1_year', w=8)
        s2_year = dynamic(features=s2_year, discount=0.95, name='sales2_year', w=6)
        m1_year = dynamic(features=m1_year, discount=0.99, name='macro_year', w=10)
        #e1_year = dynamic(features=e1_year,discount=0.95,name='eff_year',w=6)
        if enable_sales:
            drm_year = dlm(trainset1_year['cost'])+autoReg(degree=locality, name='ar2', w=5)+linear_trend_year+m1_year+s0_year+s1_year+s2_year
        else:    
            drm_year = dlm(trainset1_year['cost'])+autoReg(degree=locality, name='ar2', w=5)+linear_trend_year+m1_year#+s0_year+s1_year+s2_year
        drm_year.fit()
        testset_year = pro_sales.loc[pro_sales.wm_yr_wk_nbr>=start_test_date].head(pred_weeks)
        sales0test = testset_year['ratio'].head(pred_weeks).values.tolist()
        s0test = [[x] for x in sales0test]
        sales1test = testset_year['ratio_1'].head(pred_weeks).values.tolist()
        s1test = [[x] for x in sales1test]
        sales2test = testset_year['ratio_2'].head(pred_weeks).values.tolist()
        s2test = [[x] for x in sales2test]
        features_year = {'sales0_year':s0test,'sales1_year':s1test, 'sales2_year':s2test,'macro_year':m1test}
        (predictMean_year, predictVar_year) = drm_year.predictN(N=pred_weeks, date=drm_year.n-1,featureDict=features_year)
        weeklist = []
        p1 = np.exp(-r*(abs(end_train_date-start_of_this_year-52)))
        p2 = 1-p1
        for k in range(pred_weeks):
            weeklist.append(wm_nbr_add(start_test_date,2*k))
                    
        if res.shape[0] == 0:
            res['wm_yr_wk_nbr'] = weeklist
            res['club'] = pd.Series(club*np.ones(pred_weeks),index=res.index)
            res['yhat'] = pd.Series(p1*np.asarray(predictMean)+p2*np.asarray(predictMean_year),index=res.index)
        else:
            tmp = pd.DataFrame()
            tmp['wm_yr_wk_nbr'] = weeklist
            tmp['club'] = pd.Series(club*np.ones(pred_weeks),index=tmp.index)
            tmp['yhat'] = pd.Series(p1*np.asarray(predictMean)+p2*np.asarray(predictMean_year),index=tmp.index)
            res = pd.concat([res,tmp],axis = 0)
    return res

def mape(y, yhat):
    try:
        return np.mean(np.abs(y-yhat)/np.abs(y))*100
    except:
        return np.nan
    
    
def mixedmodel(df_sales,end_train_date,start_test_date,calendar,punched_df,start_of_this_year,enable_sales,r=0.06,locality=10):
    '''
    this is the final output model
    DLM + Prophet. Idea is Prophet for long trend and DLM responsible for local trend
    weight between two models needs more concern, so far grid seraching
    scat_plan data is not used at this moment, should be useful in the future.
    return type: dataframe.
    ''' 
    df_propor_PR_ts = predict_proportion(calendar,df_sales,end_train_date, start_test_date)
    df_sales_fc = df_propor_PR_ts.groupby('wm_yr_wk_nbr')['total_sales_across'].mean().reset_index()
    res = estimate_and_predict_dlm_PR(calendar,df_propor_PR_ts, punched_df, end_train_date,start_test_date, start_of_this_year,enable_sales)
    ttmp = estimate_and_predict_prophet_PR(calendar,punched_df, end_train_date, start_test_date,daily_view = False)
    ttmp=pd.merge(left=ttmp, right=calendar, how='left', left_on='ds', right_on= 'calendar_date').drop(['calendar_date','ds'],axis=1)
    ##res = ttmp
    res.columns = ['wm_yr_wk_nbr_local','club_local','yhat_local'] #dlm served mainly as local model
    res = pd.merge(left=res,right=ttmp,left_on=['club_local','wm_yr_wk_nbr_local'],right_on=['club','wm_yr_wk_nbr'],validate='1:1').drop(['club_local','wm_yr_wk_nbr_local'],axis=1)
    p1 = np.exp(-r*(end_train_date-start_of_this_year))
    p2 = 1-p1
    res['yhat_mixed'] = p1*res['yhat'] +p2*res['yhat_local']
    return res
def testpunched():
    '''
    backtesting function for the punched_model
    return type: mape list
    
    '''
    end_train_date= [201811,201815,201819,201823,201827]
    start_test_date= [201813,201817,201821,201825,201829]
    exclude = [0,4041,4925,6279]
    #df_sales_total['total_sales'] = df_sales_total['total_sales'].rolling(2).mean()
    mape_comp = []
    enable_sales = False
    #calendar= pd.read_csv('wm_yr_wk_ref.csv',parse_dates=[0], names=['calendar_date','wm_yr_wk_nbr'],header=0)
    for i in range(len(start_test_date)):
        start_of_this_year = math.floor(end_train_date[i]/100)
        raw= pd.read_csv('./data/sap_data_20181213.csv')
        storemap= pd.read_csv('./storemap.csv', usecols=[1,2,3,4,5])
        scat_op_club= pd.read_csv('./scat_op_club.csv', usecols=[2] )
        [punched_df,residual_worked_df,retro_df,holiday_df,lump_df,severance_df,calendar] = prep_data(raw,storemap,scat_op_club)
        
        punched_df_train = punched_df.loc[punched_df.wm_yr_wk_nbr<= end_train_date[i]]
        punched_df_test = punched_df.loc[punched_df.wm_yr_wk_nbr>= start_test_date[i]]
        df_sales = pd.read_csv('./data/PR_sales.csv')
        df_sales = df_sales[df_sales.WM_YEAR_WK_NBR.isin(punched_df.wm_yr_wk_nbr.unique().tolist())]
        df_sales = df_sales.sort_values(by ='WM_YEAR_WK_NBR' )
        df_sales = df_sales.loc[df_sales.WM_YEAR_WK_NBR <= 201813]
        df_sales_train = df_sales.loc[df_sales.WM_YEAR_WK_NBR <= end_train_date[i]]
        df_sales_test = df_sales.loc[df_sales.WM_YEAR_WK_NBR >= start_test_date[i]]
        df_sales_total = df_sales.groupby('WM_YEAR_WK_NBR')['total_sales'].sum()
        df_sales_total = df_sales_total.reset_index()
        df_sales_total = df_sales_total.sort_values(by='WM_YEAR_WK_NBR')
        res = mixedmodel(df_sales_train,end_train_date[i],start_test_date[i],calendar,punched_df_train,start_of_this_year,enable_sales)
        club_ls = res.club.unique()
        punched_pro_test = punched_df_test.groupby(['club_nbr','posting_date'])['cost'].sum()
        punched_pro_test.column = ['total_punched_wg']
        punched_pro_test = punched_pro_test.reset_index()
        punched_pro_test = pd.merge(left=punched_pro_test, right= calendar, how='left', left_on='posting_date', right_on= 'calendar_date').drop('calendar_date',axis=1)
        mape_mixed = []
        for club in club_ls:
            pro_club = punched_pro_test[punched_pro_test.club_nbr.isin([club])]
            testset = pro_club.loc[pro_club.wm_yr_wk_nbr>=start_test_date[i]]
            testset = testset.head(8)
            res_club = res[res.club.isin([club])]
            
            mape_mixed.append(mape(np.asarray(testset['cost']),np.asarray(res_club['yhat_mixed'])))
        mape_comp.append(mape_mixed) 
    return mape_comp    
        
        

def compeltetable(calendar,residual_worked_df,punched_df):
    '''
    it was mainly used when modeling other categories of worked wages.
    for each individual club, some categories does not show up in SAP data
    return the dataframe with same format as punched_df with filled zero.
    compatiable to use prophet_estimation above. 
    '''
    club_ls = punched_df.club_nbr.unique()
    start_date = punched_df.sort_values(by=['wm_yr_wk_nbr']).iloc[0,1]
    start_wk = punched_df.sort_values(by=['wm_yr_wk_nbr']).iloc[0,-2]
    end_date = punched_df.sort_values(by=['wm_yr_wk_nbr']).iloc[-1,1]
    ref_ls = residual_worked_df.rf_doc_num.unique()
    res = pd.DataFrame()
    for ref in ref_ls:
        for club in club_ls:
            tmp = residual_worked_df.loc[(residual_worked_df.club_nbr==club)&(residual_worked_df.rf_doc_num==ref)]
            curdate = start_date
            while curdate <= end_date:
                if tmp.loc[tmp.posting_date==curdate].shape[0] == 0:
                    wk = calendar.loc[calendar.calendar_date==curdate].iloc[0,1]
                    tmp = tmp.append({'rf_doc_num':50,'posting_date':curdate,'cost':0,'retail':0,'club_nbr':club,'country':'PR','city_name':'S','state_prov_code':'PR','postal_code':0,'wm_yr_wk_nbr':wk,'date_pd':curdate},ignore_index=True)
                curdate = curdate+timedelta(days=14)
            if res.shape[0] == 0:
                res = tmp
            else:
                res = pd.concat([res,tmp],axis = 0)
    return res
    
def testresidualworked(residual_worked_df):
    '''
    main function for other categories of worked wages
    return: per club.
    '''
    
    end_train_date= [201845]#,201815,201819,201823,201827]
    start_test_date= [201847]#,201817,201821,201825,201829]
    start_of_this_year = 201800
    exclude = [0,4041,4925,6279]
    [punched_df,residual_worked_df,retro_df,holiday_df,lump_df,severance_df] = prep_data()
    df_sales = pd.read_csv('./data/PR_sales.csv')
    df_sales_total = df_sales.groupby('WM_YEAR_WK_NBR')['total_sales'].sum()
    df_sales_total = df_sales_total.reset_index()
    df_sales_total['total_sales'] = df_sales_total['total_sales'].rolling(2).mean()
    mape_comp = []
    calendar= pd.read_csv('wm_yr_wk_ref.csv',parse_dates=[0], names=['calendar_date','wm_yr_wk_nbr'],header=0)
    residual_worked_df = compeltetable(calendar,residual_worked_df,punched_df)
    print(residual_worked_df.groupby(['posting_date','club_nbr'])['cost'].sum())
    for i in range(len(start_test_date)-1,len(start_test_date)):
        ttmp = estimate_and_predict_prophet_PR(calendar,residual_worked_df, end_train_date[i], start_test_date[i],daily_view = False)