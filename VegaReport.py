import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from pandas.tseries.offsets import BDay
import eikon
from functools import reduce
import MarketData
sns.set()

glb = globals() #using globals() function to iterate through multiple dataframes

#implied correlation function deriving from three implied volatilities using fx implied vol triangle
def implied_correlation(v1,v2,v3):
    return np.sqrt((v1**2+v2**2-v3**2)/(2*v1*v2))


#setting api key for eikon
eikon.set_app_key('de5fd998085b4279b6379598d1503c3796432a5a')

#csv reports file location
file_location = 'C:/Users/user/Downloads'

#setting valuation date and reference rate (value date -1BD)
value_date = MarketData.value_date
ref_date = pd.to_datetime(value_date) - BDay(1)
ref_date = datetime.datetime.strftime(ref_date,format='%Y-%m-%d')


#reading CSV files and creating dataframes (setting Maturity as index and renaming columns)
raw_file = pd.read_csv(fr'{file_location}/Vega.csv',skiprows=3,header=0)
raw_file_eurils = pd.read_csv(fr'{file_location}/VegaEURILS.csv',skiprows=3,header=0,encoding='cp1252')
raw_file_eurusd = pd.read_csv(fr'{file_location}/VegaEURUSD.csv',skiprows=3,header=0,encoding='cp1252')

file_names= [ raw_file,raw_file_eurils,raw_file_eurusd]
df_list = ['df' , 'df_eurils' , 'df_eurusd']

#creating multiple dataframes (df , df_eurils, df_eurusd) for Vega , VegaEURILS, and VegaEURUSD
for name, dataframe in zip(file_names,df_list):
    glb[dataframe] = pd.DataFrame(name)
    glb[dataframe].set_index('Maturity', inplace=True)
    if dataframe != 'df':
        glb[dataframe] = glb[dataframe].rename(columns={'Unnamed: 2':'Vega in Euro'})
    else:
        glb[dataframe] = glb[dataframe].rename(columns={'Unnamed: 2':'Vega in $'})


#removing non_numeric columns from multiple dataframes
non_int =[]

for col in df:
    if df[col].dtype !='int64':
        non_int.append(col)

for dataframe in df_list:
    glb[dataframe] = glb[dataframe].drop(columns=non_int)


#merging usdils,eurils, and eurusd dataframes to aggregate values
data_frames = [df,df_eurils,df_eurusd]
df_merged = reduce(lambda left,right:pd.merge(left,right,on=['Maturity'],how='outer'),data_frames)
df_merged['agg_vega'] = df_merged.sum(axis=1)

convert_rate = eikon.get_timeseries('eur=',fields='Close',start_date=ref_date,end_date=value_date,interval='daily')

#splitting the market data response into three dataframes

data_usdils = MarketData.data[MarketData.data.columns[0:10]].columns
data_eurils = MarketData.data[MarketData.data.columns[10:20]].columns
data_eurusd = MarketData.data[MarketData.data.columns[20:30]].columns


#creaeting market data dfs for daily change
vol_dfs = ['df_usdils_vols','df_eurils_vols','df_eurusd_vols']

for dataframe in vol_dfs:
    glb[dataframe] = MarketData.glb[dataframe]


#creating dataframes of the 1-day change in values

df_change = pd.DataFrame(df_usdils_vols) - pd.DataFrame(df_usdils_vols).shift(1)
df_change_eurils = pd.DataFrame(df_eurils_vols) - pd.DataFrame(df_eurils_vols).shift(1)
df_change_eurusd = pd.DataFrame(df_eurusd_vols) - pd.DataFrame(df_eurusd_vols).shift(1)

one_day_move = MarketData.one_day_move
one_day_move_eurils = MarketData.one_day_move_eurils
one_day_move_eurusd = MarketData.one_day_move_eurusd

#creating correlation dataframe to calculate the implied eur-usd-ils correlation and correlation change
df_corr = pd.DataFrame(columns=df.index,index=df_eurusd_vols.index)

for col in range(0,len(df_corr.columns)):
    for row in range(0,len(df_corr.index)):
        df_corr.iloc[row,col] = implied_correlation(df_usdils_vols.iloc[row,col],
                                                    df_eurusd_vols.iloc[row,col],
                                                    df_eurils_vols.iloc[row,col])

df_corr = df_corr.astype('float64')
corr_change = pd.DataFrame(df_corr) - pd.DataFrame(df_corr).shift(1)

df_corr =df_corr.transpose()
corr_change = corr_change.transpose()


#Plotting outputs:
#1. Vega expousre and 1-day vol change across the different pairs (+aggregated book)
#2. Implied correlation and correlation 1-day change

fig, axes = plt.subplots(2,2,figsize=(10,10))

sns.barplot(x=df.index,y=df['Vega in $'],ax=axes[0,0])
axes[0,0].set_xticks(np.arange(0,len(df.index)),df.index[0:],rotation=45,fontsize=8)
axes[0,0].grid(False)

ax1 = axes[0,0].twinx()
sns.lineplot(x=data_usdils,y=one_day_move,ls='dashed',
             lw=1.5,color='navy',ax=ax1,label='vol change')
yabs_max =abs(max(ax1.get_ylim(),key=abs))
ax1.set_ylim(ymin = -yabs_max,ymax=yabs_max)
ax1.set_ylabel('1-day Vol Change',fontsize=9)
ax1.grid(False)

axes[0,0].set_title('USDILS Vega Buckets',fontsize=9)

sns.barplot(x=df_eurils.index,y=df_eurils['Vega in Euro'],ax=axes[0,1])
axes[0,1].set_xticks(np.arange(0,len(df.index)),df_eurils.index[0:],rotation=45,fontsize=7)
ax3_ymax = abs(max(axes[0,1].get_ylim(),key=abs))
axes[0,1].set_ylim(ymin=-ax3_ymax,ymax=ax3_ymax)
axes[0,1].set_title('EURILS Vega Buckets',fontsize=9)

ax2 = axes[0,1].twinx()
sns.lineplot(x=data_eurils,y=one_day_move_eurils,
             lw=1.5,ls='dashed',color='navy', ax=ax2)
yabs_max_eur =abs(max(ax2.get_ylim(),key=abs))
ax2.set_ylim(ymin = -yabs_max_eur,ymax=yabs_max_eur)
ax2.set_ylabel('1-day Vol Change',fontsize=9)
ax2.grid(False)

sns.barplot(x=df_eurusd.index,y=df_eurusd['Vega in Euro'],ax=axes[1,0])
axes[1,0].set_xticks(np.arange(0,len(df_eurusd.index)),df_eurusd.index[0:],rotation=45,fontsize=7)

ax3 = axes[1,0].twinx()
sns.lineplot(x=data_eurusd,y=one_day_move,
             ls='dashed',lw=1.5,color='navy',ax=ax3)
yabs_max =abs(max(ax3.get_ylim(),key=abs))
ax3.set_ylim(ymin = -yabs_max,ymax=yabs_max)
ax3.set_ylabel('1-day Vol Change',fontsize=9)
ax3.grid(False)

axes[1,0].set_title('EURUSD Vega Buckets',fontsize=9)
sns.barplot(x=df_merged.index,y=df_merged['agg_vega'],ax=axes[1,1])
axes[1,1].set_xticks(np.arange(0,len(df_merged.index)),df_merged.index[0:],rotation=45,fontsize=7)
axes[1,1].set_title('Aggregated Vega Buckets',fontsize=9)

for i in range(0,2):
    for j in range(0,2):
        axes[i,j].grid(False)
        axes[i, j].set_ylabel('Vega in $', fontsize=9)
        axes[i, j].tick_params(axis='both', labelsize=8)
        axes[i,j].set_xlabel('')
        axes[i,j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        yabs_max = abs(max(axes[i,j].get_ylim(),key=abs))
        axes[i,j].set_ylim(-yabs_max,yabs_max)

plt.suptitle(f'Vega Exposure Breakdown by expiries for {value_date}', fontsize=12)

plt.tight_layout(pad=3)

plt.savefig(f'Vega_Report_{value_date}.pdf',format='pdf')
plt.show()

#Implied correlation plots
fig = plt.figure(figsize=(9,9))
g = sns.barplot(x= df_corr.index, y= df_corr.iloc[:,0],alpha=0.5,ci=68,errcolor='.2',capsize=0.2)
g1 = sns.barplot(x=df_corr.index, y=df_corr.iloc[:,1],ci=90, palette='hls',alpha=0.9)
g1.bar_label(g1.containers[0],size=9,label_type='center', fmt='%.2f')

g1.set_ylabel('Implied Correlation', fontsize=10)
g1.set_xlabel('Maturity', fontsize =12)
g1.set_xticks(np.arange(0,len(df.index)),df.index,rotation=45,fontsize=8)
ax2 = plt.twinx()
ax2.plot(corr_change.iloc[:,1],color='navy',lw=2,ls='--')
y_max = abs(max(ax2.get_ylim(),key=abs))
ax2.set_ylim(ymin=-y_max,ymax=y_max)
ax2.set_ylabel('1-day Change in implied Correlation', fontsize=10)

plt.xticks(np.arange(0,len(df_corr)),df.index,rotation=45,fontsize=8)

plt.title('EUR-USD-ILS Implied Correlation by tenors', fontsize=12)
plt.grid(False)
ax2.grid(False)
plt.tight_layout(pad=2)
plt.show()


#Aggregating Vega P&Ls across the different books (including aggergated values) to broadcast to
# P&L attribution report

df['vol change'] = one_day_move.values
df_eurils['vol change'] = one_day_move_eurils.values
df_eurusd['vol change'] = one_day_move_eurusd.values

df['vega pnl'] = df['Vega in $']*df['vol change']
df_eurils['vega pnl'] = df_eurils['Vega in Euro']*df_eurils['vol change']
df_eurusd['vega pnl'] = df_eurusd['Vega in Euro']*df_eurusd['vol change']

total_vega_pnl = pd.DataFrame(df['vega pnl']).sum()
total_vega_pnl_eur = pd.DataFrame(df_eurils['vega pnl']).sum() * convert_rate.iloc[-1].values
tota_vega_pnl_eurusd = pd.DataFrame(df_eurusd['vega pnl']).sum() * convert_rate.iloc[-1].values

