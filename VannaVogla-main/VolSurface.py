import numpy as np
import math
from scipy.stats import norm
from datetime import timedelta
import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import heston
import plotly.graph_objects as go
sns.set()

#Strike Convetion
delta_call_put = []
delta_range = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.49,0.49,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
delta_names = ['5d Put','10d Put', '15d Put', '20d Put', '25d Put', '30d Put','35d Put','40d Put','45d Put','50d Put',
               '50d Call', '45d Call','40d Call','35d Call','30d Call','25d Call','20d Call','15d Call','10d Call','5d Call']
call_put = ['PUT','PUT','PUT','PUT','PUT','PUT','PUT','PUT','PUT','PUT','CALL','CALL','CALL','CALL','CALL','CALL','CALL','CALL','CALL','CALL']

for delta,cp in zip(delta_range,call_put):
    delta_strike = [delta,cp]
    delta_call_put.append(delta_strike)


# Converting dates diff to year fraction
def T(expiry_date, value_date):
    date2 = datetime.datetime.strptime(expiry_date, '%Y-%m-%d')
    date1 = datetime.datetime.strptime(value_date, '%Y-%m-%d')
    return ((date2 - date1) / timedelta(days=365))

#Defining BS d1/d2 Parameters
def intVal(S, fwd, K):
    return math.log((S + fwd) / K)

def sqrtTime(v, expiry_date, value_date):
    return math.sqrt(T(expiry_date,value_date))*v

def timeVal(v, expiry_date, value_date):
    return (v * v * T(expiry_date, value_date)) / 2 / sqrtTime(v, expiry_date, value_date)


class VolSurface(object):

    def __init__(self, S, K, fwd, expiry_date, value_date, v , Vatm, RR, BF, CallPut, delta,pos,Notional):
        self.S = S                          #Spot Price
        self.K = K                          #Strike Price
        self.fwd = fwd                      #Fwd/swap pts
        self.expiry_date = expiry_date      #Expiry date in YYYY-MM-DD format
        self.value_date= value_date         #Value date in YYYY-MM-DD format
        self.v= v                           #Volatility in 0.x format
        self.Vatm= Vatm                     #ATM volatility
        self.RR = RR                        #Risk Reversal volatility
        self.BF = BF                        #Butterfly volatility
        self.CallPut = CallPut              #Call/Put toggle
        self.delta = delta                  #RR/BB corresponded delta
        self.pos = pos
        self.Notional = Notional

    #Defining Main Volatility Points - Put/Call Strikes

    def pos(self):
        if self.upper()=='CALL':
            return 1
        else:
            return -1

    def nprime(x):
        return np.exp(-(x**2)/2)*1/(2*np.pi)

    def VolK(Vatm,BF,RR,CallPut):
        if CallPut.upper()=='CALL':
            return Vatm+(BF/100)+(RR/100)/2
        else:
            return Vatm+(BF/100)-(RR/100)/2
    #print(VolK(0.105,0.25,0.25,'Call'))
    #print(VolK(0.105,0.25,0.25,'Put'))

    # Defining d1,d2 of BS
    def d1(S, K, expiry_date, value_date, fwd, v):
        return (intVal(S,fwd,K)+ timeVal(v,expiry_date,value_date))

    def d2(S, K, expiry_date, value_date, fwd, v):
        return VolSurface.d1(S, K, expiry_date, value_date, fwd, v) - sqrtTime(v,expiry_date,value_date)

    def BS_Vega(S, K, expiry_date, value_date, fwd, v):
        return (norm.pdf(VolSurface.d1(S, K, expiry_date, value_date, fwd, v)) * (S + fwd) * math.sqrt(T(expiry_date, value_date)) * 0.01) / S

    def BS_Vanna(S,K,expiry_date,value_date,fwd,v,BuySell,Notional=1):
        return VolSurface.pos(BuySell)*np.sqrt(T(expiry_date,value_date))*VolSurface.nprime(VolSurface.d1(S,K,expiry_date,value_date,fwd,v))*\
               (1-VolSurface.d1(S,K,expiry_date,value_date,fwd,v))*Notional/10



    def BS_Volga(S,fwd,K,v,expiry_date,value_date,BuySell):
        t=T(expiry_date,value_date)

        vega_up = VolSurface.BS_Vega(S,K,expiry_date,value_date,fwd,v+0.01,BuySell,Notional=1)
        vega_down = VolSurface.BS_Vega(S,K,expiry_date,value_date,fwd,v-0.01,BuySell,Notional=1)
        return (vega_up-vega_down)/2/100


# Derive strike from delta
def GetStrikeFromDelta(S,fwd,v,expiry_date,value_date,CallPut,delta):
        if CallPut.upper() == 'CALL':
            return float((S+fwd)*np.exp(-norm.ppf(delta)*v*np.sqrt(T(expiry_date,value_date))+((v**2)/2)*T(expiry_date,value_date)))
        else:
            return float((S+fwd)*np.exp(norm.ppf(delta)*v*np.sqrt(T(expiry_date,value_date))+((v**2)/2)*T(expiry_date,value_date)))


# the sensitivity of vol to change of spot price
def dVoldSpot(S, fwd,expiry_date,value_date,Vatm,RR,BF,delta):

 CallStrike=GetStrikeFromDelta(S,fwd,VolSurface.VolK(Vatm,RR,BF,'CALL'),expiry_date,value_date,'CALL',0.25)
 PutStrike=GetStrikeFromDelta(S,fwd,VolSurface.VolK(Vatm,RR,BF,'Put'),expiry_date,value_date,'Put',0.25)

 dVoldSpotU=((VolSurface.VolK(Vatm,RR,BF,'Call')-Vatm)/(math.log(CallStrike)/(S+fwd)))
 dVoldSpotD=((VolSurface.VolK(Vatm,RR,BF,'Put')-Vatm)/(math.log(PutStrike)/(S+fwd)))

 return (dVoldSpotU+dVoldSpotD)/2/100




def SmilePL(S,fwd,K,expiry_date,value_date,Vatm,RR,BF,VolK,delta,BuySell):
    VolCall = VolSurface.VolK(Vatm, BF, RR, CallPut='CALL')
    VolPut = VolSurface.VolK(Vatm, BF, RR, CallPut='PUT')

    CallStrike =GetStrikeFromDelta(S,fwd,VolSurface.VolK(Vatm,RR,BF,'CALL'),expiry_date,value_date,'CALL',0.25)
    PutStrike = GetStrikeFromDelta(S,fwd,VolSurface.VolK(Vatm,RR,BF,'Put'),expiry_date,value_date,'Put',0.25)

    dVoldSpot_temp=dVoldSpot(S,fwd,expiry_date,value_date,Vatm,RR,BF,delta)
    vega= VolSurface.BS_Vega(S,K,expiry_date,value_date,fwd,VolK)
    vanna = VolSurface.BS_Vanna(S,K,expiry_date,value_date,fwd,VolK,BuySell)
    volga = VolSurface.BS_Volga(S,fwd,K,VolK,expiry_date,value_date,BuySell)


    if (S+fwd)>K:
        return (dVoldSpot_temp*(vega+vanna*(math.log(S+fwd)/CallStrike)+volga*VolCall))/(math.log(S+fwd)/CallStrike)
    else:
        return (dVoldSpot_temp*(vega+vanna*(math.log(S+fwd)/PutStrike)+volga*VolPut))/(math.log(S+fwd)/PutStrike)

def strangle_adjustment(Vatm,RR,BF,ref_spot,expiry_date,value_date,fwd):
    Vput = Vatm - (RR/100)*0.5 +BF/100
    Vcall = Vatm + (RR/100)*0.5 +BF/100

    Kput = GetStrikeFromDelta(ref_spot,fwd,Vput,expiry_date,value_date,'PUT',0.25)
    Kcall = GetStrikeFromDelta(ref_spot,fwd,Vcall,expiry_date,value_date,'CALL',0.25)

    vega_adj = (Vput*VolSurface.BS_Vega(ref_spot,Kput,expiry_date,value_date,fwd,Vatm)+Vcall*VolSurface.BS_Vega(ref_spot,Kcall,expiry_date,value_date,fwd,Vatm))/\
               (VolSurface.BS_Vega(ref_spot,Kput,expiry_date,value_date,fwd,Vatm)+VolSurface.BS_Vega(ref_spot,Kcall,expiry_date,value_date,fwd,Vatm))
    return vega_adj - Vatm


#print(SmilePL(100,0,99,'2016-12-16','2016-11-11',0.105,-0.25,0.25,0.106,0.25))

# Volatility Surface Interpolation using Quadratic spline method (Vanna-Volga)
def VannaVolgaImpliedVol(S,fwd,K, expiry_date,value_date,Vatm,RR,BF,delta):

     K1= GetStrikeFromDelta(S,fwd,VolSurface.VolK(Vatm,RR,BF,'PUT'),expiry_date,value_date,'PUT',delta)
     K2= (S+fwd)*math.exp((Vatm**2)/2*T(expiry_date,value_date))
     K3= GetStrikeFromDelta(S,fwd,VolSurface.VolK(Vatm,RR,BF,'CALL'),expiry_date,value_date,'CALL',delta)


     S1= Vatm - (RR/100)*0.5+BF/100
     S3 = Vatm + (RR/100)*0.5 + BF/100


     y1= (math.log(K2/K)*math.log(K3/K))/(math.log(K2/K1)*math.log(K3/K1))
     y2= (math.log(K/K1)*math.log(K3/K))/(math.log(K2/K1)*math.log(K3/K2))
     y3= (math.log(K/K1)*math.log(K/K2))/(math.log(K3/K1)*math.log(K3/K2))




     d1 = (math.log(S / K) + (Vatm * Vatm * T(expiry_date, value_date) / 2)) / (
                 np.sqrt(T(expiry_date, value_date)) * Vatm)
     d2 = d1 - Vatm * np.sqrt(T(expiry_date, value_date))
     d1d2 = d1 * d2

     p= y1*S1+y2*Vatm+y3*S3-Vatm
     q= y1*VolSurface.d1(S,K1,expiry_date,value_date,fwd,S1)*VolSurface.d2(S,K1,expiry_date,value_date,fwd,S1)*(S1-Vatm)**2+\
        y3*VolSurface.d1(S,K3,expiry_date,value_date,fwd,S3)*VolSurface.d2(S,K3,expiry_date,value_date,fwd,S3)*((S3-Vatm)**2)



     return Vatm+(-Vatm+math.sqrt(Vatm**2+d1*d2*(2*Vatm*p+q)))/d1d2

underlying = 'USD/JPY'
ref_spot = 113.38
value_date = '2021-11-27'

expiry_dates = np.array(['2021-12-29','2022-01-29','2022-03-27','2022-05-27','2022-06-27','2022-09-27','2022-11-27'])
vol_atm = np.array([0.0893,0.089,0.083,0.082,0.08,0.08,0.079])
fwd_pts = np.array([-0.021,-0.1199,-0.1507,-.22,-0.2797,-.38,-0.697])
RR_25d = np.array([-1.75,-1.688,-1.555,-1.55,-1.412,-1.32,-1.244])
BF_25d_1vol = np.array([0.208,0.238,0.245,0.25,0.265,0.28,0.316])

BF_25d_2vol = []
market_data_temp,market_data = [],[]
vv_vol = []

for expiry, vatm, fwd, rr, bf in zip(expiry_dates,vol_atm,fwd_pts,RR_25d,BF_25d_1vol):
    data = [expiry,vatm,fwd,rr,bf]
    market_data_temp.append(data)
    for data in market_data_temp:
        bfly = strangle_adjustment(data[1], data[3], data[4], ref_spot, data[0], value_date, data[2]) * 100
    BF_25d_2vol.append(bfly)

for expiry, vatm, fwd, rr, bf in zip(expiry_dates,vol_atm,fwd_pts,RR_25d,BF_25d_2vol):

    data = [expiry,vatm,fwd,rr,bf]
    market_data.append(data)


for md in market_data:
    vv = [VannaVolgaImpliedVol(ref_spot,fwd=md[2],
                                  K=GetStrikeFromDelta(ref_spot,md[2],md[1],md[0],value_date,strike[1],abs(strike[0])),
                                  expiry_date=md[0],
                                  value_date=value_date,
                                  Vatm=md[1],RR=md[3],BF=md[4],delta=0.25) for strike in delta_call_put]
    vv_vol.append(vv)


vv_vol = np.array(vv_vol).reshape((len(market_data),len(delta_range)),order='c')

volSurface = pd.DataFrame(vv_vol,index=expiry_dates,columns=delta_names)

fig = go.Figure(data=[go.Surface(z=volSurface.values,y=expiry_dates,x=delta_names,opacity=0.8,colorscale='balance')])
fig.update_layout(title = f'{underlying} VolSurface',width=1000,height=1000,title_x=0.5,title_y=0.9,margin=dict(t=10))
fig.update_annotations(align='center')
fig.show()



