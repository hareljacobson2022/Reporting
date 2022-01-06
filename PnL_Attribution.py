import datetime
import VegaReport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import BDay
from matplotlib.ticker import StrMethodFormatter
import eikon
sns.set()


#setting eiko api key
eikon.set_app_key('de5fd998085b4279b6379598d1503c3796432a5a')

#setting file location
file_location = 'C:/Users/user/Downloads'

#setting valuation date, reference date for previous valuation and extracting date features
value_date = VegaReport.value_date
ref_date = pd.to_datetime(value_date) - BDay(1)

day,month,year = ref_date.day, ref_date.month,ref_date.year

if int(day) < 10:
    day = str(day).zfill(2)

if int(month) < 10:
    month = str(month).zfill(2)

ref_date = datetime.datetime.strftime(ref_date,format='%Y-%m-%d')

#defining time delta
def time_delta(date1,date2):
    date1 = datetime.datetime.strptime(date1,'%Y-%m-%d')
    date2 = datetime.datetime.strptime(date2,'%Y-%m-%d')
    return pd.to_numeric(abs(date1 - date2).days,downcast='integer')


#reading raw csv files and creating dataframes
raw_file = pd.read_csv(fr'{file_location}/Greeks.csv',skiprows=3,header=0,encoding='cp1252')
raw_delta_file = pd.read_csv(fr'{file_location}/delta_{year}_{month}_{day}_22_00.csv',header=None,encoding='cp1252')

df = pd.DataFrame(raw_file)
df_delta = pd.DataFrame(raw_delta_file)

#specifing headers and books to filter df_delta, setting index column and filtering the dataframe
delta_headers = ['Date','Book','Ccys','delta']
trading_books = ['Exotic','iopthedg','VanFxFx']
df_delta.columns = delta_headers

df.set_index('Ccys',inplace=True)
df_delta.set_index('Book',inplace=True)
df_delta = df_delta[df_delta.index.isin(trading_books)]

#removing non_numeric columns from the dataframe
non_int =[]

for col in df:
    if df[col].dtype !='float64':
        non_int.append(col)

df = df.drop(columns=non_int)
df = df.rename(columns={'Unnamed: 3': 'Delta',
                        'Unnamed: 5': 'Gamma',
                        'Unnamed: 7':  'Vega',
                        'Unnamed: 9': 'Theta',
                        'Unnamed: 11': 'Rho'})

#aggregating delta by currency pairs
df = df.groupby('Ccys').sum()
df_delta = df_delta.groupby('Ccys').sum()


#removing ILS/ILS data from the dataframe
for item in df.index:
    if item[0:3] == item[4:7]:
        df = df.drop(item)

#creating tickers' list to retrieve data from eikon
rics =[]

for ccy in df.index:
    if ccy[0:3]=='USD':
        ric = ccy[4:7]+'='
    elif ccy[4:7]=='USD':
        ric = ccy[0:3]+'='
    elif ccy[0:3]==ccy[4:7]:
        pass
    else:
        ric = ccy[0:3]+ccy[4:7]+'=R'
    rics.append(ric)

unique_list = []

for item in rics:
    if item not in unique_list:
        unique_list.append(item)

rics = unique_list

#requesting data from eikon into dataframe
data = eikon.get_timeseries(rics,fields='Close',start_date=ref_date,end_date=value_date,interval='daily')


#creating returns' dataframe
df_rtn = np.log(pd.DataFrame(data)/pd.DataFrame(data).shift(1))
df_rtn = df_rtn.iloc[-1]


#calcuating daily P&L based on greeks and the underlying movement (under taylor expension):
# delta = (delta+delta(t-1))/2 * spot move
# gamma = (gamma /2)* abs(spot move)
# theta = theta * (days change from ref_date)
# vega = retreived from VegaReport if USD/ILS, EUR/ILS, or EUR/USD. otherwise, ignored
# rho = rho * base ccy move (assumed no move for now)

pnl =[]
for col in df.columns:
    for rate,index in zip(df_rtn,df.index):
        if col == 'Delta':
            pnl_greek = ((df[col][index]+df_delta['delta'][index])/2)*rate
        elif col == 'Gamma':
            pnl_greek = df[col][index]*abs(rate)/2
        elif col == 'Theta':
            pnl_greek = df[col][index]*time_delta(ref_date,value_date)
        elif col =='Rho':
            pnl_greek = df['Rho'][index]*0.001
        else:
            pass
        pnl.append(pnl_greek)

pnl = np.array(pnl).reshape((len(df.index),len(df.columns)),order='f')
pnl = pd.DataFrame(pnl,index=df.index,columns=df.columns)


for ccy in df.index:
    if ccy == 'USD/ILS':
        pnl['Vega'][ccy] = VegaReport.total_vega_pnl
    elif ccy=='EUR/ILS':
        pnl['Vega'][ccy] = VegaReport.total_vega_pnl_eur
    elif ccy=='EUR/USD':
        pnl['Vega'][ccy] = VegaReport.tota_vega_pnl_eurusd
    elif ccy=='GBP/ILS':
        pnl['Vega'][ccy] = df['Vega'][ccy] *0.1
    else:
        pass


#creating aggregated P&L from all sub currency pairs

agg_pnl = pnl.sum()

#modifing the USD/ILS,EUR/ILS, and EUR/USD dataframes for plotting

usdils_pnl = pd.melt(pnl[pnl.index=='USD/ILS'])
eurils_pnl = pd.melt(pnl[pnl.index=='EUR/ILS'])
eurusd_pnl = pd.melt(pnl[pnl.index=='EUR/USD'])

#plotting P&L attribution

fig ,axes = plt.subplots(2,2,figsize=(9,9))

sns.barplot(x=pnl.columns, y = agg_pnl.values,ax=axes[0,0])
sns.barplot(x=usdils_pnl.variable,y=usdils_pnl.value,ax=axes[0,1])
sns.barplot(eurils_pnl.variable,y=eurils_pnl.value,ax=axes[1,0])
sns.barplot(eurusd_pnl.variable,y=eurusd_pnl.value,ax=axes[1,1])

y_max_ax00 = abs(max(axes[0,0].get_ylim(),key=abs))
y_max_ax01 = abs(max(axes[0,1].get_ylim(),key=abs))
y_max_ax10 = abs(max(axes[1,0].get_ylim(),key=abs))
y_max_ax11 = abs(max(axes[1,1].get_ylim(),key=abs))

axes[0,0].set_ylim(ymin=-y_max_ax00,ymax=y_max_ax00)
axes[0,1].set_ylim(ymin=-y_max_ax01,ymax=y_max_ax01)
axes[1,0].set_ylim(ymin=-y_max_ax10,ymax=y_max_ax10)
axes[1,1].set_ylim(ymin=-y_max_ax11,ymax=y_max_ax11)

axes[0,0].set_title(f'Options Desk 1-day P&L attribution by Greeks. P&L : {agg_pnl.sum(): ,.0f}',fontsize=8)
axes[0,1].set_title(f'USD/ILS 1-day P&L attribution by Greeks. P&L : {usdils_pnl.sum().value : ,.0f}', fontsize=8)
axes[1,0].set_title(f'EUR/ILS 1-day P&L attribution by Greeks. P&L : {eurils_pnl.sum().value: ,.0f}', fontsize=8)
axes[1,1].set_title(f'EUR/USD 1-day P&L attribution by Greeks. P&L : {eurusd_pnl.sum().value: ,.0f}', fontsize=8)

for i in range(0,2):
    for j in range(0,2):
        axes[i,j].set_ylabel('Attribution in $',fontsize=8)
        axes[i,j].set_xlabel('Options Greek',fontsize=8)
        axes[i, j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.suptitle(f'Option Desk 1-day P&L attribution for {value_date}', fontsize=12)
plt.tight_layout(pad=3)

plt.savefig(f'PnL_Attribution_{value_date}.pdf',format='pdf')
plt.show()

