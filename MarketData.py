import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
import eikon

glb = globals()

#implied correlation function deriving from three implied volatilities using fx implied vol triangle
def implied_correlation(v1,v2,v3):
    return np.sqrt((v1**2+v2**2-v3**2)/(2*v1*v2))


#setting api key for eikon
eikon.set_app_key('de5fd998085b4279b6379598d1503c3796432a5a')

#csv reports file location
file_location = 'C:/Users/user/Downloads'

#setting valuation date and reference rate (value date -1BD)
value_date = '2022-01-06'
ref_date = pd.to_datetime(value_date) - BDay(1)
ref_date = datetime.datetime.strftime(ref_date,format='%Y-%m-%d')



#retreiving vol data from Reuters:
#using usdils tickers' format to create eurils and eurusd format.
#combining all three lists into one list and sending one combined request to create data output (df)
usdils_vol = ['ILSONO=','ILSSWO=','ILS2WO=R','ILS1MO=','ILS2MO=','ILS3MO=','ILS6MO=','ILS9MO=','ILS1YO=','ILS2YO=']
eurils_vol,eurusd_vol = [],[]

for ticker in usdils_vol:
    eur_vol = 'EUR'+ticker
    eurusd = 'EUR'+ticker[3:7]
    eurils_vol.append(eur_vol)
    eurusd_vol.append(eurusd)

args = (usdils_vol,eurils_vol,eurusd_vol)

vol_list = np.concatenate(args)
tickers=[]
for item in vol_list:
    tickers.append(str(item))


data = eikon.get_timeseries(tickers,fields='Close',start_date=ref_date,end_date=value_date,interval='daily')
convert_rate = eikon.get_timeseries('eur=',fields='Close',start_date=ref_date,end_date=value_date,interval='daily')

#splitting the market data response into three dataframes
df_usdils_vols = data[data.columns[0:10]]
df_eurils_vols = data[data.columns[10:20]]
df_eurusd_vols = data[data.columns[20:30]]

#creating dataframes of the 1-day change in values
df_change = pd.DataFrame(df_usdils_vols) - pd.DataFrame(df_usdils_vols).shift(1)
df_change_eurils = pd.DataFrame(df_eurils_vols) - pd.DataFrame(df_eurils_vols).shift(1)
df_change_eurusd = pd.DataFrame(df_eurusd_vols) - pd.DataFrame(df_eurusd_vols).shift(1)

one_day_move = df_change.iloc[-1]
one_day_move_eurils = df_change_eurils.iloc[-1]
one_day_move_eurusd = df_change_eurusd.iloc[-1]

