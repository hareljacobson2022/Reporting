import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.ticker import StrMethodFormatter
import Files_Loader
import sys
from MarketData import value_date
import eikon
from numba import jit

sys.path.insert(0,'./VannaVogla-main/')
import VannaVolga_ImpliedVol as VV



ccys = ['ILS=', 'EURILS=']
data = eikon.get_timeseries(rics=ccys,fields='CLOSE',start_date=value_date,end_date=value_date,interval='daily')

usdils= float(data['ILS='].iloc[-1])
eurils = float(data['EURILS='].iloc[-1])

glb = globals()

value_date = '2022-01-18'
file_location = 'C:/Users/user/Downloads'

raw_usdils_file = Files_Loader.raw_file_vega
raw_eurils_file = Files_Loader.raw_file_vega_eurils
usdils_market_data = pd.read_csv(fr'{file_location}/usdils_vol.csv')
eurils_market_data = pd.read_csv(fr'{file_location}/eurils_vol.csv')

file_names = [raw_usdils_file, raw_eurils_file,usdils_market_data,eurils_market_data]
df_names = ['df_usdils', 'df_eurils','df_usdils_md','df_eurils_md']

for name, dataframe in zip(file_names[0:2],df_names[0:2]):
    glb[dataframe] = pd.DataFrame(name)
    glb[dataframe] = glb[dataframe].rename(columns={'Unnamed: 0': 'strike'})
    glb[dataframe].set_index('strike', inplace=True)


for name,dataframe in zip(file_names[2:4],df_names[2:4]):
    glb[dataframe] = pd.DataFrame(name)
    glb[dataframe].set_index('Expiry',inplace=True)
    glb[dataframe].index = pd.to_datetime(glb[dataframe].index)
    glb[dataframe]['expiry'] = glb[dataframe].index


headers=['1wk','2wks','1m','2m','3m','6m','9m','1y']

headers_to_remove =[]
headers_to_keep=[]
for i in np.arange(2,18,2):
    name = 'Unnamed: '+str(i)
    headers_to_remove.append(name)

non_int_usdils , non_int_eurils =[],[]


for col in df_usdils:
    if df_usdils[col].dtype !='int64':
        non_int_usdils.append(col)

for col in df_eurils:
    if df_eurils[col].dtype !='int64':
        non_int_eurils.append(col)

df_usdils = df_usdils.drop(columns=non_int_usdils)
df_eurils = df_eurils.drop(columns=non_int_eurils)

for dataframe in df_names[0:2]:
    for name,new_name in zip(headers_to_remove,headers):
        glb[dataframe] = glb[dataframe].rename(columns = {name:new_name})


cols =[]

cols = [col.strftime('%Y-%m-%d') for col in df_usdils_md['expiry']]

df_usdils_skew = pd.DataFrame(columns=cols,index=df_usdils.index)
df_eurils_skew = pd.DataFrame(columns=cols,index=df_eurils.index)



usdils_spot = usdils
eurils_spot =eurils

df_usdils_skew = df_usdils_skew[(df_usdils_skew.index > 0.7 *usdils_spot) & (df_usdils_skew.index < 1.3 *usdils)]
df_eurils_skew =df_eurils_skew[(df_eurils_skew.index > 0.7 * eurils_spot) & (df_eurils_skew.index < 1.3 * eurils)]



df_usdils_md['expiry'] = df_usdils_md['expiry'].dt.strftime('%Y-%m-%d')
df_eurils_md['expiry'] = df_eurils_md['expiry'].dt.strftime('%Y-%m-%d')

market_data = df_usdils_md.to_numpy()
eur_market_data = df_eurils_md.to_numpy()

usdils_strikes  = df_usdils_skew.index.to_numpy()
eurils_strikes = df_eurils_skew.index.to_numpy()



for row in market_data:
    for k in usdils_strikes:
        df_usdils_skew[row[3]][k] = VV.VannaVolga(S=usdils_spot, K=k,fwd=0,
                                                      expiry_date=row[3],value_date=value_date,
                                                      v_atm=row[0],RR=row[1],BF=row[2],
                                                      CallPut='CALL',delta=0.25,BuySell='BUY',deltaBase=True).Get_dVol_dRR()

for row in eur_market_data:
    for k in eurils_strikes:
        df_eurils_skew[row[3]][k] = VV.VannaVolga(S=eurils_spot, K=k,fwd=0,
                                                     expiry_date=row[3],value_date=value_date,
                                                     v_atm=row[0],RR=row[1],BF=row[2],
                                                     CallPut='CALL',delta=0.25,BuySell='BUY',deltaBase=True,atm_conv='ATMF').Get_dVol_dRR()


df_usdils_skew = df_usdils_skew.mul(df_usdils.values).astype('float64')
df_eurils_skew = df_eurils_skew.mul(df_eurils.values).astype('float64')

ref_spot, ref_spot_eur = usdils_spot, eurils_spot
strike_range = 0.20
increment = 0.05
min_strike, min_strike_eurils = ref_spot * np.exp(-strike_range/2) , \
                                ref_spot_eur * np.exp(-strike_range/2)
max_strike, max_strike_eurils = ref_spot * np.exp(strike_range/2), \
                                ref_spot_eur * np.exp(strike_range/2)
strike_bin , strike_bin_eurils = np.around(np.arange(min_strike,max_strike,increment),3) , \
                                 np.around(np.arange(min_strike_eurils,max_strike_eurils,increment),3)

df_usdils_strikes = df_usdils_skew.groupby(pd.cut(df_usdils_skew.index,strike_bin)).sum()
df_eurils_strikes = df_eurils_skew.groupby(pd.cut(df_eurils_skew.index,strike_bin_eurils)).sum()

df_usdils_strikes_sum = df_usdils_strikes.sum(axis=0)
df_eurils_strikes_sum = df_eurils_strikes.sum(axis=0)

fig, axes = plt.subplots(2,2,figsize=(10,10))

sns.heatmap(df_usdils_strikes,annot=True, fmt=',.0f', linewidth=0.1, annot_kws={'size':7},
            vmin=df_usdils_strikes.values.min(), vmax=df_usdils_strikes.values.max(),center=0,cmap='coolwarm_r',cbar=False, ax=axes[0,0])

sns.heatmap(df_eurils_strikes,annot=True, fmt=',.0f', linewidth=0.1, annot_kws={'size':7},
            vmin=df_usdils_strikes.values.min(), vmax=df_usdils_strikes.values.max(),center=0,cmap='coolwarm_r',cbar=False, ax=axes[0,1])

sns.barplot(x=df_usdils_strikes_sum.index, y=df_usdils_strikes_sum.values, ax=axes[1,0])
sns.barplot(x=df_eurils_strikes_sum.index, y=df_eurils_strikes_sum.values , ax=axes[1,1])

axes[0,0].set_yticks(np.arange(0,len(df_usdils_strikes.index)),df_usdils_strikes.index,fontsize=8)
axes[0,0].set_xticks(np.arange(0,len(df_usdils_strikes.columns)),df_usdils_strikes.columns,fontsize=8,rotation=45)
axes[0,0].set_title('USD/ILS RR Sensitivity',fontsize=10)

axes[0,1].set_yticks(np.arange(0,len(df_eurils_strikes.index)),df_eurils_strikes.index,fontsize=8)
axes[0,1].set_xticks(np.arange(0,len(df_eurils_strikes.columns)),df_eurils_strikes.columns,fontsize=8,rotation=45)
axes[0,1].set_title('EUR/ILS RR Sensitivity',fontsize=10)

for i in range(0,2):
    axes[1, i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[1, i].tick_params(axis='both', labelsize=8)
    axes[1, i].tick_params(axis='x', rotation=45)

axes[1,0].set_title('USD/ILS RR Sensitivity by tenors',fontsize=8)
axes[1,1].set_title('EUR/ILS RR Sensitivity by tenors',fontsize=8)
axes[1,0].set_ylabel('$ sensitivity for 0.1% change in RR',fontsize=8)
axes[1,1].set_ylabel('EUR sensitivity for 0.1% change in RR',fontsize=8)


plt.tight_layout(pad=3)
plt.savefig('Skew_Exposure.pdf',format='pdf')
plt.show()