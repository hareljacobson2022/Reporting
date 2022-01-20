import pandas as pd
from pandas.tseries.offsets import BDay
import MarketData
import datetime
from numpy import where

value_date = MarketData.value_date
ref_date = pd.to_datetime(value_date) - BDay(1)

day,month,year,weekday = pd.to_datetime(value_date).day, \
                         pd.to_datetime(value_date).month,\
                         pd.to_datetime(value_date).year, \
                         pd.to_datetime(value_date).weekday()

#adjust the day,month length in the raw data-files when day,month < 10
day = where(int(day)<10,str(day).zfill(2),str(day))
month = where(int(month) < 10, str(month).zfill(2),str(month))

ref_date = datetime.datetime.strftime(ref_date,format='%Y-%m-%d')

file_location = 'C:/Users/user/Downloads'

#parsing Strike360 files for EURILS and USDILS
raw_file = pd.read_csv(fr'{file_location}/Strikes360.csv',skiprows=4,header=0, encoding='cp1252')
raw_file_vega = pd.read_csv(fr'{file_location}/StrikesVega360.csv',skiprows=4,header=0, encoding='cp1252')
raw_file_eurils = pd.read_csv(fr'{file_location}/Strikes360EURILS.csv',skiprows=4,header=0, encoding='cp1252')
raw_file_vega_eurils = pd.read_csv(fr'{file_location}/StrikesVega360EURILS.csv',skiprows=4,header=0, encoding='cp1252')

#parsing DetailedVegaReport raw files

if weekday==4:
    raw_usdils_file = pd.read_csv(fr'{file_location}/USDILS_vol_node_point_analysis_{year}_{month}_{day}_13_00.csv',skiprows=3,header=0)
    raw_eurils_file = pd.read_csv(fr'{file_location}/EURILS_vol_node_point_analysis_{year}_{month}_{day}_13_00.csv',skiprows=3,header=0)
else:
    raw_usdils_file = pd.read_csv(fr'{file_location}/USDILS_vol_node_point_analysis_{year}_{month}_{day}_17_30.csv',
                                  skiprows=3, header=0)
    raw_eurils_file = pd.read_csv(fr'{file_location}/EURILS_vol_node_point_analysis_{year}_{month}_{day}_17_30.csv',
                                  skiprows=3, header=0)

#parsing VegaReport raw files
raw_vega_usdils = pd.read_csv(fr'{file_location}/Vega.csv',skiprows=3,header=0)
raw_vega_eurils = pd.read_csv(fr'{file_location}/VegaEURILS.csv',skiprows=3,header=0,encoding='cp1252')
raw_vega_eurusd = pd.read_csv(fr'{file_location}/VegaEURUSD.csv',skiprows=3,header=0,encoding='cp1252')

#parse PnL_Attribution raw files
if weekday==4:
    raw_pnl_attribution = pd.read_csv(fr'{file_location}/Greeks_{year}_{month}_{day}_13_00.csv',skiprows=3,header=0,encoding='cp1252')
else:
    raw_pnl_attribution = pd.read_csv(fr'{file_location}/Greeks_{year}_{month}_{day}_17_30.csv', skiprows=3, header=0,
                                      encoding='cp1252')