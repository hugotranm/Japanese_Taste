# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:54:13 2017

@author: Tran-Pro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import openpyxl
import matplotlib
import altair as alt


air_reserve = pd.read_csv('C:\\Users\\Tran-Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\air_reserve.csv',
                          parse_dates = True, index_col = 'reserve_datetime')
air_visit = pd.read_csv('C:\\Users\\Tran-Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\air_visit_data.csv', 
                        parse_dates = True, index_col = 'visit_date')
air_store = pd.read_csv('C:\\Users\\Tran Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\air_store_info.csv') 
hpg_reserve = pd.read_csv('C:\\Users\\Tran Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\hpg_reserve.csv')
hpg_store = pd.read_csv('C:\\Users\\Tran Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\hpg_store_info.csv')
store_id_relation = pd.read_csv('C:\\Users\\Tran Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\store_id_relation.csv')
test_set = pd.read_csv('C:\\Users\\Tran Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\sample_submission.csv')
date_info = pd.read_csv('C:\\Users\\Tran-Pro\\Dropbox\\Project for fun\\Restaurant Recruiting\\Data\\date_info.csv', 
                        parse_dates = True, index_col = 'calendar_date')

#Looking at reserve date vs visitors
air_reserve_rd = air_reserve.drop(['air_store_id','visit_datetime'], axis = 1)
air_reserve_rd['2016-01-02'].sum()
air_rday = air_reserve.resample('D').sum()
air_rday_series = pd.Series(data = air_rday.reserve_visitors, index = air_rday.index) 


fig1 = plt.figure(figsize = (15,5))
ax1 = fig1.add_subplot(1,1,1)
ax1.set_xlabel('Date Reserved')
ax1.set_ylabel('Reserve Visitors')
ax1.set_title('Visits by Reservation')
ax1.plot(air_rday['reserve_visitors'], color = 'steelblue', label = 'Visitors')
ax1.legend(loc = 'upper left')


#Looking at actual visitors without reservation
air_visitors = air_visit.resample('D').sum()
air_visit.groupby(air_visit.air_store_id).sum()

fig2 = plt.figure(figsize = (15,5))
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('Date Visited')
ax2.set_ylabel('Visitors')
ax2.set_title('Visits General')
ax2.plot(air_visitors['visitors'], color = 'steelblue', label = 'Visitors')
ax2.legend(loc = 'upper left')

#### Combined both plots
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (20,10))
axes[0].plot('reserve_visitors', data = air_rday)
axes[0].set_title('Visits by Reservation')
axes[1].plot('visitors', data = air_visitors)
axes[1].set_title('General Visits', y = -0.1)
plt.subplots_adjust(wspace=0, hspace=0)


###Now Decomposing. 
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
reserve_log = np.log(air_rday)
general_log = np.log(air_visitors)
reserve_log.dropna(inplace=True)


decomposition_re = seasonal_decompose(reserve_log, freq = 14)
reserve_trend = decomposition_re.trend
reserve_seasonal = decomposition_re.seasonal
reserve_residual = decomposition_re.resid

figu, decom_axes = plt.subplots(nrows = 4, ncols = 1, figsize = (22,14))
decom_axes[0].plot('reserve_visitors', data = air_rday)
decom_axes[0].set_title('Original')
decom_axes[0].legend(loc='best')
decom_axes[1].plot('reserve_visitors', data = reserve_trend)
decom_axes[1].set_title('Trend')
decom_axes[1].legend(loc='best')
decom_axes[2].plot(reserve_seasonal, label='Seasonality')
decom_axes[2].set_title('Seasonality')
decom_axes[2].legend(loc='best')
decom_axes[3].plot(reserve_residual, label='Residuals')
decom_axes[3].set_title('Residuals')
decom_axes[3].legend(loc='best')
plt.subplots_adjust(wspace=0.22, hspace=0.22)

##### and test stationary
#Dropd NaN
reserve_residual.dropna(inplace = True)
residual_ts = pd.Series(reserve_residual['reserve_visitors'].values, index = reserve_residual['reserve_visitors'].index)
test_resid = reserve_residual['reserve_visitors'].values 
from statsmodels.tsa.stattools import adfuller
adfuller(test_resid)[0:4]
type(adfuller(test_resid)[4])
print(adfuller(test_resid))
def stationary(ts): 
    #rolling first
    rollingmean = ts.rolling(window = '14D').mean()
    rollingstd = ts.rolling(window = '14D').std()
    
    #graph to identify trend
    fig, axes= plt.subplots(nrows = 3, ncols = 1, figsize = (20,10))
    axes[0].plot(ts, color = 'steelblue', label = 'Original')
    axes[0].set_title('Original')
    axes[0].legend(loc='best')
    axes[1].plot(rollingmean, color = 'red', label = 'Moving Average')
    axes[1].set_title('Moving Average')
    axes[1].legend(loc='best')
    axes[2].plot(rollingstd, color = 'black', label = 'Variance')
    axes[2].set_title('Variance')
    axes[2].legend(loc='best')
    plt.show()
    
    #Dicky-Fuller test
    print('Results:')
    DFresult = pd.Series(adfuller(ts)[0:4], index = ['Test Statistic','p-value'
                       ,'Lags Used','Number of Observations Used'])
    for key, value in adfuller(ts)[4].items():
        DFresult['Critical Value %s' % key] = value
    print(DFresult)
    
stationary(residual_ts)

#Percent Difference between reserve vs actual visitors
pct_arr = (air_rday.reserve_visitors.values / air_visitors.visitors.values)*100
pct_ts = pd.Series(pct_arr, index = air_rday.index, name = 'Differences in %')
pct_mean = pct_ts.rolling(window='14D').mean()

# % in Histogram
pct_ts.hist(figsize = (12,8), color = 'lightcoral', label = 'Percent',rwidth = 0.85)

# % in Line chart
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (20,10))
axes[0].plot(pct_mean, color = 'steelblue', label = '% difference 14D')
axes[0].legend(loc = 'best')
axes[0].set_title('Smoothed % Difference')
axes[1].plot(pct_ts, color = 'salmon', label = '% difference')
axes[1].legend(loc = 'best')
axes[1].set_title('Difference b/t Reserve vs Actual', y = -0.25)
plt.subplots_adjust(wspace=0, hspace=0)

######Join the Holiday in 
# Include Day_Name
mapped_day = pd.Series(air_rday.index.weekday, index = air_rday.index).map(pd.Series('Mon Tue Wed Thu Fri Sat Sun'.split()))
air_rday['day_name'] = mapped_day.values
air_rday.groupby(['day_name']).sum()
air_rday_name = air_rday.set_index(['day_name'])
type(air_rday_name.groupby(['day_name']).sum())

# Add Holiday Flag 
date = date_info[:'4/22/2017']
air_rday['holiday_flg'] = date['holiday_flg'].values

# Try mark holiday vertical spans on Matplotlib
import matplotlib.dates as mdates
import datetime as dt
holiday_only = date[(date.holiday_flg == 1)]
holiday_only.index

fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (40,10))
axes[0].plot('reserve_visitors', data = air_rday)
axes[0].set_title('Visits by Reservation')
axes[0].axvspan(*mdates.date2num([x for x in holiday_only.index]), color = 'gray', alpha = 0.5)
axes[0].axvspan(*mdates.datestr2num(['1/1/2017', '1/3/2017']), color = 'gray', alpha = 0.5)

axes[1].plot('visitors', data = air_visitors)
axes[1].set_title('General Visits', y = -0.1)
axes[1].axvspan(*mdates.datestr2num(['1/1/2016', '1/3/2016']), color = 'gray', alpha = 0.5)
plt.subplots_adjust(wspace=0, hspace=0)




ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

air_rday.plot()
plt.show()


air_reserve.describe()
air_reserve.quantile()
air_reserve.info()

air_visit.describe()
air_visit.quantile()
air_visit.info()

air_store.describe()
air_store.quantile()
air_store.info()

air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
air_reserve.reserve_datetime
air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
air_reserve.visit_datetime
air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
air_visit.visit_date

hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
hpg_reserve.reserve_datetime
hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve.visit_datetime

#missing data
air_reserve.isnull().sum()
air_store.isnull().sum()
air_visit.isnull().sum()

timeit air_visit.isnull().sum()
timeit air_reserve.isnull().sum()

#visualizing
import altair as alt

figure1 = plt.figure(figsize=(50,50))
visitors_plot = figure1.add_subplot(2,2,1)

visitors_plot.plot(air_reserve.visit_datetime, air_reserve.reserve_visitors, color ='blue')
plt.show()

plt.plot(air_reserve.visit_datetime, air_reserve.reserve_visitors, color ='blue')
air_reserve.visit_datetime.values.year










