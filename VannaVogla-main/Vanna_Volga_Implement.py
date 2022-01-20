import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from BlackScholes import BlackScholes
from VannaVolga_function import GetStrikeFromDeltaPA , GetStrikeFromDelta , VolK
from VannaVolga_ImpliedVol import VannaVolga
sns.set()

delta_call_put =[]
delta_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49, 0.49, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
                   0.15, 0.1, 0.05]
delta_names = ['5d Put', '10d Put', '15d Put', '20d Put', '25d Put', '30d Put', '35d Put', '40d Put', '45d Put',
                   '50d Put',
                   '50d Call', '45d Call', '40d Call', '35d Call', '30d Call', '25d Call', '20d Call', '15d Call',
                   '10d Call', '5d Call']
call_put = ['PUT', 'PUT', 'PUT', 'PUT', 'PUT', 'PUT', 'PUT', 'PUT', 'PUT', 'PUT', 'CALL', 'CALL', 'CALL',
                'CALL', 'CALL', 'CALL', 'CALL', 'CALL', 'CALL', 'CALL']

for delta, cp in zip(delta_range, call_put):
    delta_strike = [delta, cp]
    delta_call_put.append(delta_strike)

def plot_smile(S,fwd,v,RR,BF,expiry_date,value_date,delta,deltaBase=True,atm_conv='DN'):
    vv_vol = []

    for strike in delta_call_put:
        K = GetStrikeFromDelta(S, fwd, v, expiry_date, value_date, strike[1], strike[0])
        vol = VannaVolga(S,K,fwd,expiry_date,value_date,v,RR,BF,'CALL',0.25,deltaBase,atm_conv).GetImpliedVol()
        vv_vol.append(vol)

    plt.plot(vv_vol)
    plt.xticks(np.arange(0,len(delta_names)),delta_names,rotation=45,fontsize=8)
    plt.show()

plot_smile(114.29,-0.170,6.26,-0.77,0.225,'2022-03-22','2021-12-22',0.25,True,'DN')

def plot_surface(S,fwd_curve,v_atm,RR_term,BF_term,expiries,value_date,delta,deltaBase=True,atm_conv='DN'):

    market_data=[]
    vol_surface =[]

    for expiry,fwd,vol,rr,bf in zip(expiries,fwd_curve,v_atm,RR_term,BF_term):
        data = [expiry,fwd,vol,rr,bf]
        market_data.append(data)

    for term in market_data:
        vol = [VannaVolga(S,
                          GetStrikeFromDelta(S=S,fwd=term[1],v=term[2],
                                             expiry_date=term[0],value_date=value_date,CallPut=strike[1],
                                             delta=strike[0]),
                          fwd=term[1],expiry_date=term[0],value_date=value_date,
                          v_atm=term[2],RR=term[3],BF=term[4],CallPut='CALL',delta,
                          deltaBase=deltaBase,atm_conv=atm_conv).GetImpliedVol() for strike in delta_call_put]
        vol_surface.append(vol)



    vol_surface = np.array(vol_surface).reshape((len(expiries),len(delta_call_put)),order='f')
    df = pd.DataFrame(vol_surface,index=expiries,columns=delta_names)

    fig = go.Figure(data=[go.Surface(z=df.values,
                                     x=delta_names,
                                     y=expiries, colorscale='Viridis')])

    fig.update_layout(autosize=True,width=1000,height=1000,scene=dict(xaxis_title='delta strikes',
                                                      yaxis_title='expiries',
                                                      zaxis_title='volatility',))

    fig.show()

expiries = ['2021-12-29','2022-01-04','2022-01-22','2022-02-22','2022-03-22','2022-06-22','2022-09-22','2022-12-22']
fwd_curve = [0 ,-0.034,-0.038,-0.048,-0.07,-0.106,-.16,-0.210 ]
atm_curve = [4.8,5.07,5.67,6.02,6.25,6.48,6.59,6.71]
RR_curve = [-0.433,-0.66,-0.71,-0.76,-0.85,-0.888,-0.92,-0.92]
BF_curve = [0.16,0.18,0.2,0.203,0.23,0.255,0.285,0.305]

plot_surface(114.2,fwd_curve,atm_curve,RR_curve,BF_curve,expiries,'2021-12-20',0.25)

prob_surface,prob_flat_vol =[] , []
ref_spot = 12.475
fwd = 0.42
expiry_date = '2022-03-20'
value_date = '2021-12-20'
atm_vol= 74.67
RR = 10.28
BF = 3.27
strike_range = np.linspace(0.1,2.5)*ref_spot

for strike in strike_range:
    vol = VannaVolga(ref_spot,strike,fwd,expiry_date,value_date,atm_vol,RR,BF,'CALL',0.25,atm_conv='DN').GetImpliedVol()
    prob = BlackScholes(ref_spot,strike,fwd,vol,expiry_date,value_date,'CALL','BUY',1).dVdK()
    prob_flat = BlackScholes(ref_spot,strike,fwd,atm_vol,expiry_date,value_date,'CALL','BUY',1).dVdK()
    prob_surface.append(prob)
    prob_flat_vol.append(prob_flat)

plt.plot(prob_surface)
plt.plot(prob_flat_vol,color='r',ls='--')
plt.show()