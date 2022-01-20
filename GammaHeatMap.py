import eikon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pylab import *
import seaborn as sns
import MarketData , Files_Loader
sns.set()

glb= globals()
#defining the straddle breakeven
def std_be(vol,t):
    return 0.8 * vol * np.sqrt(t)


value_date = MarketData.value_date

ccys = ['ILS=', 'EURILS=']
data = eikon.get_timeseries(rics=ccys,fields='CLOSE',start_date=value_date,end_date=value_date,interval='daily')

usdils= float(data['ILS='].iloc[-1])
eurils = float(data['EURILS='].iloc[-1])

#retreiving files from Files_Loader and creating dataframes

raw_file = Files_Loader.raw_file
raw_file_vega = Files_Loader.raw_file_vega
raw_file_eurils = Files_Loader.raw_file_eurils
raw_file_vega_eurils = Files_Loader.raw_file_vega_eurils

file_names = [raw_file, raw_file_vega , raw_file_eurils , raw_file_vega_eurils]
df_names = ['df', 'df_vega', 'df_eurils', 'df_vega_eurils']

for name, dataframe in zip(file_names,df_names):
    glb[dataframe] = pd.DataFrame(name)
    glb[dataframe] = glb[dataframe].rename(columns={'Unnamed: 0':'strike'})
    glb[dataframe].set_index('strike',inplace=True)


#setting headers and time to expiry (in years terms)
headers=['1wk','2wks','1m','2m','3m','6m','9m','1y']
t = [1/52,2/52,1/12,2/12,3/12,6/12,9/12,1]

headers_to_remove,headers_to_keep =[] , []

[headers_to_remove.append('Unnamed: '+str(i)) for i in np.arange(2,18,2)]

non_int,non_int_vega =[] , []
non_int_eur , non_int_vega_eur = [],[]


for col in df:
    if df[col].dtype !='int64':
        non_int.append(col)

for col in df_vega:
    if df_vega[col].dtype !='int64':
        non_int_vega.append(col)

for col in df_eurils:
    if df_eurils[col].dtype !='int64':
        non_int_eur.append(col)

for col in df_vega_eurils:
    if df_vega_eurils[col].dtype !='int64':
        non_int_vega_eur.append(col)

df = df.drop(columns=non_int)
df_vega = df_vega.drop(columns=non_int_vega)
df_eurils = df_eurils.drop(columns=non_int_eur)
df_vega_eurils = df_vega_eurils.drop(columns=non_int_vega_eur)


for name,new_name in np.column_stack((headers_to_remove,headers)):
    df = df.rename(columns={name : new_name})
    df_vega = df_vega.rename(columns={name: new_name})
    df_eurils = df_eurils.rename(columns={name: new_name})
    df_vega_eurils = df_vega_eurils.rename(columns = {name: new_name})



#setting ref_spot , strike_range, and increments
ref_spot, ref_spot_eur = usdils, eurils
strike_range = 0.20
increment = 0.05
min_strike, min_strike_eurils = ref_spot * np.exp(-strike_range/2) , \
                                ref_spot_eur * np.exp(-strike_range/2)
max_strike, max_strike_eurils = ref_spot * np.exp(strike_range/2), \
                                ref_spot_eur * np.exp(strike_range/2)
strike_bin , strike_bin_eurils = np.around(np.arange(min_strike,max_strike,increment),3) , \
                                 np.around(np.arange(min_strike_eurils,max_strike_eurils,increment),3)


#grouping by strikes and splitting to bins
df_strikes,df_vega_strikes,df_vega1 = df.groupby(pd.cut(df.index,strike_bin)).sum() , \
                             df_vega.groupby(pd.cut(df_vega.index,strike_bin)).sum(),\
                                      df_vega.groupby('strike').sum()

df_eurils_strikes , \
df_eurils_vega_strikes , \
df_eurils_vega = df_eurils.groupby(pd.cut(df_eurils.index,strike_bin_eurils)).sum() , \
                                                              df_vega_eurils.groupby(pd.cut(df_vega_eurils.index,strike_bin_eurils)).sum(), \
                                                              df_vega_eurils.groupby('strike').sum()

#summing gamma by time buckets
df_gamma_buckets = df.sum()
df_eurils_gamma_bucket = df_eurils.sum()

#calculating skew exposure (vega above ref spot - vega below ref spot)
vega_puts, vega_calls = df_vega1[df_vega1.index <ref_spot].sum() , \
                        df_vega1[df_vega1.index > ref_spot].sum()

skew_exposure = vega_calls - vega_puts

eurils_vega_puts , eurils_vega_calls = df_vega_eurils[df_vega_eurils.index < ref_spot_eur].sum() , \
                                       df_vega_eurils[df_vega_eurils.index > ref_spot_eur].sum()

eur_skew_exposure = eurils_vega_calls - eurils_vega_puts

#generating breakeven threshold from implied atm vol curve and expiries
usdils_vols = MarketData.df_usdils_vols.iloc[-1]
eurils_vols = MarketData.df_eurils_vols.iloc[-1]
breakeven, eurils_breakeven =[] , []

for vol,expiry in zip(usdils_vols,t):
    std_neg  =  - std_be(vol,expiry)
    std_pos = std_be(vol,expiry)
    breakeven.append([std_neg, std_pos])


for vol, expiry in zip(eurils_vols,t):
    std_neg = -std_be(vol,expiry)
    std_pos = std_be(vol, expiry)
    eurils_breakeven.append([std_neg, std_pos])

#plotting strikes heatmap for gamma, vega + time buckets
#plotting USD/ILS report
fig, axes = plt.subplots(2,2,figsize=(12,12))

sns.heatmap(df_strikes,annot=True, fmt=',', linewidth=0.1, annot_kws={'size':7},
            vmin=df_strikes.values.min(), vmax=df_strikes.values.max(),center=0,cmap='coolwarm_r',
            ax=axes[0,0],cbar=False)

ax2 = axes[0,0].twinx()
ax2.plot(breakeven[0], color='black',lw=3,alpha=0.5, ls='dotted')
ax2.tick_params(axis='both',labelsize=8)
ax2.grid(False)

sns.heatmap(df_vega_strikes,annot=True, fmt=',', linewidth=0.1, annot_kws={'size':7},
            vmin=df_vega_strikes.values.min(), vmax=df_vega_strikes.values.max(),center=0,cmap='coolwarm_r',
            ax=axes[0,1],cbar=False)

sns.barplot(x=headers, y=df_gamma_buckets,palette='coolwarm_r',ax=axes[1,0])
y_max = abs(max(axes[1,0].get_ylim(),key=abs))
axes[1,0].set_ylim(ymin = -y_max, ymax=y_max)
axes[1,0].set_ylabel('Gamma in $',fontsize=8)


sns.barplot(x=headers,y=skew_exposure,palette='coolwarm_r',ax=axes[1,1])
axes[1,1].set_ylabel('Vega in $',fontsize=8)
y_max = abs(max(axes[1,1].get_ylim(),key=abs))
axes[1,1].set_ylim(ymin = -y_max, ymax=y_max)

for i in range(0,2):
        axes[0,i].set_yticks(np.arange(0,len(df_strikes.index)),df_strikes.index,fontsize=8)
        axes[0,i].set_xticks(np.arange(0,len(df_strikes.columns)),df_strikes.columns,fontsize=8)

for i in range(0,2):
    axes[1, i].set_xticks(np.arange(0, len(df_gamma_buckets)), df_gamma_buckets.index, fontsize=8)
    axes[1, i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[1, i].tick_params(axis='both', labelsize=8)


axes[0,0].set_title('USD/ILS Gamma Distribution by strikes/time buckets',fontsize=9)
axes[0,1].set_title('USD/ILS Vega Distribution by strikes/time buckets',fontsize=9)
axes[1,0].set_title('USD/ILS Gamma Buckets', fontsize=9)
axes[1,1].set_title('USD/ILS Skew exposure across maturities',fontsize=9)
plt.tight_layout(pad=3)
plt.savefig(f'Gamma_&_Vega_heatmap_USDILS_{value_date}.pdf',format='pdf')
plt.show()


#plotting EUR/ILS
fig, axes = plt.subplots(2,2,figsize=(12,12))

sns.heatmap(df_eurils_strikes,annot=True, fmt=',', linewidth=0.1, annot_kws={'size':7},
            vmin=df_eurils_strikes.values.min(), vmax=df_eurils_strikes.values.max(),center=0,cmap='coolwarm_r',
            ax=axes[0,0],cbar=False)

ax2 = axes[0,0].twinx()
ax2.plot(eurils_breakeven, color='black',lw=3,alpha=0.5, ls='dotted')
ax2.tick_params(axis='both',labelsize=8)
ax2.grid(False)

sns.heatmap(df_eurils_vega_strikes,annot=True, fmt=',', linewidth=0.1, annot_kws={'size':7},
            vmin=df_eurils_vega_strikes.values.min(), vmax=df_eurils_vega_strikes.values.max(),center=0,cmap='coolwarm_r',
            ax=axes[0,1],cbar=False)

sns.barplot(x=headers, y=df_eurils_gamma_bucket,palette='coolwarm_r',ax=axes[1,0])
y_max = abs(max(axes[1,0].get_ylim(),key=abs))
axes[1,0].set_ylim(ymin = -y_max, ymax=y_max)
axes[1,0].set_ylabel('Gamma in $',fontsize=8)


sns.barplot(x=headers,y=eur_skew_exposure,palette='coolwarm_r',ax=axes[1,1])
axes[1,1].set_ylabel('Vega in $',fontsize=8)
y_max = abs(max(axes[1,1].get_ylim(),key=abs))
axes[1,1].set_ylim(ymin = -y_max, ymax=y_max)

for i in range(0,2):
        axes[0,i].set_yticks(np.arange(0,len(df_eurils_strikes.index)),df_eurils_strikes.index,fontsize=8)
        axes[0,i].set_xticks(np.arange(0,len(df_eurils_strikes.columns)),df_eurils_strikes.columns,fontsize=8)

for i in range(0,2):
    axes[1, i].set_xticks(np.arange(0, len(df_gamma_buckets)), df_eurils_gamma_bucket.index, fontsize=8)
    axes[1, i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    axes[1, i].tick_params(axis='both', labelsize=8)


axes[0,0].set_title('EUR/ILS Gamma Distribution by strikes/time buckets',fontsize=9)
axes[0,1].set_title('EUR/ILS Vega Distribution by strikes/time buckets',fontsize=9)
axes[1,0].set_title('EUR/ILS Gamma Buckets', fontsize=9)
axes[1,1].set_title('EUR/ILS Skew exposure across maturities',fontsize=9)
plt.tight_layout(pad=3)
plt.savefig(f'Gamma_&_Vega_heatmap_EURILS_{value_date}.pdf',format='pdf')
plt.show()