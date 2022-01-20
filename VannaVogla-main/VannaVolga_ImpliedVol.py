import matplotlib.pyplot as plt
import seaborn as sns
import BlackScholes as bs
from BlackScholes import BlackScholes
import VannaVolga_function as vv
import numpy as np
import math
sns.set()


class VannaVolga:

    def __init__(self,S,K,fwd,
                 expiry_date,value_date,
                 v_atm , RR , BF ,
                 CallPut,delta,
                 BuySell='BUY',Notional=1,deltaBase=True,atm_conv='DN',strike_type='rate'):
        self.S = S
        self.fwd = fwd
        self.f = (S+fwd)
        self.K = K
        self.expiry_date = expiry_date
        self.value_date = value_date
        self.t = bs.T(expiry_date,value_date)
        self.v_atm = v_atm
        self.v = v_atm/100
        self.RR = RR
        self.BF = BF
        self.CallPut = CallPut.upper()
        self.delta = delta
        self.BuySell = BuySell
        self.Notional = Notional
        self.deltaBase = deltaBase
        self.atm_conv = atm_conv
        self.strike_type = strike_type

    # Strike Convetion

    def GetImpliedVol(self):
        s, fwd , f , K , t, expiry_date,value_date, v , RR , BF ,delta ,CallPut = \
            self.S, self.fwd ,self.f , self.K ,self. t ,self.expiry_date,self.value_date ,\
            self.v_atm,self.RR , self.BF , self.delta,self.CallPut

        def d1(s,fwd,k,v,t):
            return (np.log((s+fwd)/k)+((v**2)/2)*t)/(np.sqrt(t)*v)

        def d2(s,fwd,k,v,t):
            return d1(s,fwd,k,v,t) - np.sqrt(t)*v

        BF = vv.convert_bfly(v,BF,RR)

        S1 = vv.VolK(v,BF,RR,'PUT')/100
        S2 = v/100
        S3 = vv.VolK(v,BF,RR,'CALL')/100

        K1 = np.where(self.deltaBase==True,
                      vv.GetStrikeFromDeltaPA(s,fwd,vv.VolK(v,BF,RR,'PUT'),expiry_date,value_date,'PUT',delta),
                      vv.GetStrikeFromDelta(s, fwd, vv.VolK(v, BF, RR, 'PUT'), expiry_date, value_date, 'PUT', delta)
                      )
        K2 = np.where(self.deltaBase==True,
                      np.where(self.atm_conv=='DN',
                               vv.Get_DN_Strike(s,fwd,v,expiry_date,value_date,deltaBase=True),
                               np.where(self.atm_conv=='ATMF',f,s)),
                      f * np.exp((S2 ** 2) / 2 * t))

        K3 = np.where(self.delta==True,
                      vv.GetStrikeFromDeltaPA(s,fwd,vv.VolK(v,BF,RR,'CALL'),expiry_date,value_date,'CALL',delta),
                      vv.GetStrikeFromDelta(s, fwd, vv.VolK(v, BF, RR, 'CALL'), expiry_date, value_date, 'CALL', delta))

        y1 = (math.log(K2/K)*math.log(K3/K))/(math.log(K2/K1)*math.log(K3/K1))
        y2 = (math.log(K/K1)*math.log(K3/K))/(math.log(K2/K1)*math.log(K3/K2))
        y3 = (math.log(K/K1)*math.log(K/K2))/(math.log(K3/K1)*math.log(K3/K2))

        p = y1*S1 + y2 * S2 + y3* S3 - S2
        q = y1 * d1(s,fwd,K1,S2,t)*d2(s,fwd,K3,S2,t)*((S1-S2)**2) +\
            y3 * d1(s,fwd,K3,S2,t)*d2(s,fwd,K3,S2,t)*((S3-S2)**2)
        d1d2 = d1(s,fwd,K,S2,t)*d2(s,fwd,K,S2,t)
        return (S2+(-S2+np.sqrt(S2**2+d1d2*(2*S2*p+q)))/d1d2)*100

    def GetImpliedSkew(self):
        s, fwd, f, K, t, expiry_date, value_date, v, RR, BF, delta, CallPut = \
            self.S, self.fwd, self.f, self.K, self.t, self.expiry_date, self.value_date, \
            self.v_atm, self.RR, self.BF, self.delta, self.CallPut

        if self.deltaBase==True:
            K_Call = vv.GetStrikeFromDeltaPA(S=s,fwd=fwd,v=v,expiry_date=expiry_date,value_Date=value_date,CallPut='CALL',delta=delta)
            K_Put = vv.GetStrikeFromDeltaPA(S=s,fwd=fwd,v=v,expiry_date=expiry_date,value_Date=value_date,CallPut='PUT',delta=delta)
        else:
            K_Call = vv.GetStrikeFromDelta(S=s,fwd=fwd,v=v,expiry_date=expiry_date,value_date=value_date,CallPut='CALL',delta=delta)
            K_Put = vv.GetStrikeFromDelta(S=s,fwd=fwd,v=v,expiry_date=expiry_date,value_date=value_date,CallPut='PUT',delta=delta)

        vol_call = VannaVolga(S=s,K=K_Call,fwd=fwd,
                              expiry_date=expiry_date,value_date=value_date,
                              v_atm=v,RR=RR,BF=BF,CallPut='CALL',delta=delta,deltaBase=self.deltaBase,atm_conv=self.atm_conv).GetImpliedVol()

        vol_put = VannaVolga(S=s, K=K_Put, fwd=fwd,
                              expiry_date=expiry_date, value_date=value_date,
                              v_atm=v, RR=RR, BF=BF, CallPut='CALL', delta=delta, deltaBase=self.deltaBase,
                              atm_conv=self.atm_conv).GetImpliedVol()

        return (vol_call - vol_put)/ (np.log(K_Call/K_Put))/100

    def Get_dVol_dRR(self):
        s, fwd, f, K, t, expiry_date, value_date, v, RR, BF, delta = \
            self.S, self.fwd, self.f, self.K, self.t, self.expiry_date, self.value_date, \
            self.v_atm, self.RR, self.BF, self.delta,

        delta_RR = 0.1
        RR_up = RR +delta_RR
        RR_dn = RR -delta_RR

        vol_up = VannaVolga(S=s,K=K,fwd=fwd,
                            expiry_date=expiry_date,value_date=value_date,
                            v_atm=v,RR=RR_up,BF=BF,CallPut='CALL',delta=delta,deltaBase=self.deltaBase,atm_conv=self.atm_conv).GetImpliedVol()

        vol_dn = VannaVolga(S=s,K=K,fwd=fwd,
                            expiry_date=expiry_date,value_date=value_date,
                            v_atm=v,RR=RR_dn,BF=BF,CallPut='CALL',delta=delta,deltaBase=self.deltaBase,atm_conv=self.atm_conv).GetImpliedVol()

        return ((vol_up-vol_dn)/(delta_RR*2))/10

    def Get_dVol_dBF(self):
        s, fwd, f, K, t, expiry_date, value_date, v, RR, BF, delta = \
            self.S, self.fwd, self.f, self.K, self.t, self.expiry_date, self.value_date, \
            self.v_atm, self.RR, self.BF, self.delta

        delta_BF = 0.1
        BF_up = BF + delta_BF
        BF_dn = BF - delta_BF

        vol_up = VannaVolga(S=s, K=K, fwd=fwd,
                            expiry_date=expiry_date, value_date=value_date,
                            v_atm=v, RR=RR, BF=BF_up, CallPut='CALL', delta=delta, deltaBase=self.deltaBase,
                            atm_conv=self.atm_conv).GetImpliedVol()

        vol_dn = VannaVolga(S=s, K=K, fwd=fwd,
                            expiry_date=expiry_date, value_date=value_date,
                            v_atm=v, RR=RR, BF=BF_dn, CallPut='CALL', delta=delta, deltaBase=self.deltaBase,
                            atm_conv=self.atm_conv).GetImpliedVol()

        return ((vol_up - vol_dn) / (delta_BF * 2)) / 10




    def ReVega(self):
        s, fwd, K, t, expiry_date, value_date, v, RR, BF, delta,bs,n = \
            self.S, self.fwd, self.K, self.t, self.expiry_date, self.value_date, \
            self.v_atm, self.RR, self.BF, self.delta,self.BuySell,self.Notional

        vega_volga = BlackScholes(S=s,K=K,fwd=fwd,v=v,
                                  expiry_date=expiry_date,value_date=value_date,
                                  CallPut='CALL',BuySell=bs,Notional=n).vega_volga_risk()

        dVol_dRR = VannaVolga.Get_dVol_dRR(self)

        return (vega_volga*dVol_dRR)

    def SeVega(self):
        s, fwd, K, t, expiry_date, value_date, v, RR, BF, delta, bs, n = \
            self.S, self.fwd, self.K, self.t, self.expiry_date, self.value_date, \
            self.v_atm, self.RR, self.BF, self.delta, self.BuySell, self.Notional

        vega_volga = BlackScholes(S=s, K=K, fwd=fwd, v=v,
                                  expiry_date=expiry_date, value_date=value_date,
                                  CallPut='CALL', BuySell=bs, Notional=n).vega_volga_risk()
        dVol_dBF = VannaVolga.Get_dVol_dBF(self)

        return (vega_volga*dVol_dBF)
