import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import MarketData, Files_Loader
sns.set()

glb = globals()

value_date = MarketData.value_date

#reading csv files
raw_usdils_file = Files_Loader.raw_usdils_file
raw_eurils_file = Files_Loader.raw_eurils_file

#creating dataframes:
# renaming and aranging columns headers
#dropping non_numeric columns
files = [raw_usdils_file,raw_eurils_file]
df_names = ['df_usdils','df_eurils']
agg_df =[]
column_names = {'Unnamed: 0':'Book',
                'Unnamed: 1' : 'Maturity',
                'Unnamed: 3' : 'ATM_VOL',
                'Unnamed: 5': '25_RR',
                'Unnamed: 7':'10_RR',
                'Unnamed: 9': '25_BF',
                'Unnamed: 11':'10_BF',
                'Unnamed: 13': 'Total PnL'}

for file, dataframe in zip(files,df_names):
    glb[dataframe] = pd.DataFrame(file)\
                         .iloc[:80,0:]\
                        .rename(columns=column_names)
    agg_df.append(dataframe+'_agg')


non_int=[]

for col in df_usdils.iloc[:,2:]:
    if df_usdils[col].dtype !='float64':
        non_int.append(col)

for dataframe in df_names:
    glb[dataframe] = glb[dataframe].drop(columns=non_int).iloc[:,1:]


#deleting raw data files
del raw_eurils_file,raw_usdils_file,non_int


expiry_list =[]
#Aggregating data across maturities:
#Creating tenor index (from expiry dates)
#Creating aggergated dataframes for each currency pair based on tenor index


for item in df_usdils['Maturity']:
    if item not in expiry_list:
        expiry_list.append(item)

for dataframe,agg_dataframe in zip(df_names,agg_df):
    glb[agg_dataframe] = pd.DataFrame(columns=glb[dataframe].columns[1:],index=expiry_list)


for col in df_usdils.columns[1:]:
    for term in df_usdils_agg.index:
        df_usdils_agg[col][term] = df_usdils[col][df_usdils['Maturity']==term].sum()
        df_eurils_agg[col][term] = df_eurils[col][df_eurils['Maturity']==term].sum()

#returning sum P&L for each column
total_atm_pnl,tota_eur_atm_pnl = df_usdils_agg['ATM_VOL'].sum() , df_eurils_agg['ATM_VOL'].sum()
total_rr_pnl, total_eur_rr_pnl = df_usdils_agg['25_RR'].sum() , df_eurils_agg['25_RR'].sum()
total_bf_pnl , total_eur_bf_pnl = df_usdils_agg['25_BF'].sum() , df_eurils_agg['25_BF'].sum()


#Total Desk Vega-related P&L = USD/ILS + EUR/ILS + EUR/USD
total_pnl = total_atm_pnl +total_rr_pnl +total_bf_pnl
total_eur_pnl = tota_eur_atm_pnl +total_eur_rr_pnl + total_eur_bf_pnl

x_labels = [ i for i in df_usdils_agg.index]

#Plotting data
fig, axes = plt.subplots(2,2,figsize=(10,10))


sns.barplot(x=df_usdils_agg.index, y= df_usdils_agg['ATM_VOL'], ax = axes[0,0])
sns.barplot(x=df_usdils_agg.index, y= df_usdils_agg['25_RR'], ax = axes[0,1])
sns.barplot(x=df_usdils_agg.index, y= df_usdils_agg['25_BF'], ax = axes[1,0])
sns.barplot(x=df_usdils_agg.index, y= df_usdils_agg['Total PnL'], ax = axes[1,1])

axes[0,0].set_title(f'USD/ILS ATM Vol exposure breakdown by tenors. Total exposure {total_atm_pnl: ,.0f}',fontsize=9)
axes[0,1].set_title(f'USD/ILS Skew exposure breakdown by tenors. Total exposure {total_rr_pnl: ,.0f}',fontsize=9)
axes[1,0].set_title(f'USD/ILS vol-convexity exposure breakdown by tenors. Total exposure {total_bf_pnl: ,.0f}',fontsize=9)
axes[1,1].set_title(f'USD/ILS aggegated vol exposure by tenors. Total exposure {total_pnl:,.0f}',fontsize=9)

for i in range(0,2):
    for j in range(0,2):
        axes[i,j].set_xticklabels(x_labels, rotation=45, fontsize=8)
        axes[i,j].set_ylabel('Exposure in ILS', fontsize=9)
        max_y = abs(max(axes[i,j].get_ylim(),key=abs))
        axes[i,j].set_ylim(-max_y,max_y)
        axes[i, j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[i,j].tick_params(axis='y',labelsize=9)

plt.suptitle(f'Detailed Vega-related exposures by tenors for {value_date}',fontsize=10)

plt.tight_layout(pad=3)
plt.savefig(f'Detailed_Vega_Report_{value_date}.pdf',format='pdf')
plt.show()


fig, axes = plt.subplots(2,2,figsize=(10,10))


sns.barplot(x=df_eurils_agg.index, y= df_eurils_agg['ATM_VOL'], ax = axes[0,0])
sns.barplot(x=df_eurils_agg.index, y= df_eurils_agg['25_RR'], ax = axes[0,1])
sns.barplot(x=df_eurils_agg.index, y= df_eurils_agg['25_BF'], ax = axes[1,0])
sns.barplot(x=df_eurils_agg.index, y= df_eurils_agg['Total PnL'], ax = axes[1,1])

axes[0,0].set_title(f'EUR/ILS ATM Vol exposure breakdown by tenors. Total exposure {total_eur_rr_pnl: ,.0f}',fontsize=9)
axes[0,1].set_title(f'EUR/ILS Skew exposure breakdown by tenors. Total exposure {total_eur_rr_pnl: ,.0f}',fontsize=9)
axes[1,0].set_title(f'EUR/ILS vol-convexity exposure breakdown by tenors. Total exposure {total_eur_bf_pnl: ,.0f}',fontsize=9)
axes[1,1].set_title(f'EUR/ILS aggegated vol exposure by tenors. Total exposure {total_eur_pnl:,.0f}',fontsize=9)

for i in range(0,2):
    for j in range(0,2):
        axes[i,j].set_xticklabels(x_labels, rotation=45, fontsize=8)
        axes[i,j].set_ylabel('Exposure in ILS', fontsize=9)
        max_y = abs(max(axes[i,j].get_ylim(),key=abs))
        axes[i,j].set_ylim(-max_y,max_y)
        axes[i, j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[i,j].tick_params(axis='y',labelsize=9)

plt.suptitle(f'Detailed Vega-related exposures by tenors for {value_date}',fontsize=10)

plt.tight_layout(pad=3)
plt.savefig(f'Detailed_EURILS_Vega_Report_{value_date}.pdf',format='pdf')
plt.show()