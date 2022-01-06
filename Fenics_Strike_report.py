import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
from matplotlib import cm
import eikon
sns.set()


file_location = 'C:/Users/user/Downloads'

raw_file = pd.read_csv(fr'{file_location}/StrikesUSDILS_2021_12_24_22_00.csv',skiprows=4,header=0)
df = pd.DataFrame(raw_file)
df= df.rename(columns={'Unnamed: 0' : 'strike'})
df.set_index('strike',inplace=True)

headers_to_remove =[]
headers_to_keep=[]
for i in np.arange(2,18,2):
    name = 'Unnamed: '+str(i)
    headers_to_remove.append(name)

non_int =[]

for col in df:
    if df[col].dtype !='int64':
        non_int.append(col)
for name in df.columns:
    if name not in headers_to_remove:
        headers_to_keep.append(name)

df = df.drop(columns=non_int)

for name,new_name in zip(headers_to_remove,headers_to_keep):
    df = df.rename(columns={name : new_name})


ref_spot = 3.15
strike_range = 0.05
increment = 0.02
min_strike = ref_spot * np.exp(-strike_range)
max_strike = ref_spot * np.exp(strike_range)
strike_bin = np.around(np.arange(min_strike,max_strike,increment),3)
#
df_strikes = df.groupby(pd.cut(df.index,strike_bin)).sum()
df_sum_columns = df.sum()

xpos = np.arange(df_strikes.shape[0])
ypos = np.arange(df_strikes.shape[1])
yposM,xposM = np.meshgrid(ypos,xpos)
zpos = np.zeros(df_strikes.shape).flatten()
dx = 0.5 *np.ones_like(zpos)
dy = 0.5 * np.ones_like(zpos)
dz = df_strikes.values.ravel()
#
values = np.linspace(0.2,1.7,xposM.ravel().shape[0])
color = cm.rainbow(values)
#
fig = plt.gcf()
fig.set_size_inches(18.5,10.5,forward=True)
fig.set_dpi(50)
ax = fig.add_subplot(111,projection='3d')
ax.bar3d(xposM.ravel(),yposM.ravel(),zpos,dx,dy,dz,color=color,alpha=0.8)
ax.set_yticklabels(headers_to_keep,fontsize=14)
ax.set_xticklabels(strike_bin,fontsize=14)
plt.subplots_adjust(left=0.0,right=1.0,bottom=0.0,top=1.0)
plt.show()
#
# sns.barplot(x=df.columns,y=df_sum_columns)
# plt.xticks(rotation=45,fontsize=9)
# plt.xlabel('Expiry')
# plt.ylabel('Gamma')
# plt.title('Gamma Exposure per Expiry')
# plt.show()


