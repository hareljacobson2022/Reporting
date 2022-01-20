import BlackScholes
import numpy as np
from scipy.stats import norm
from scipy import optimize

def VolK(v_atm,BF,RR,CallPut):
    return v_atm +BF +BlackScholes.cp(CallPut.upper())*RR/2

def GetStrikeFromDelta(S,fwd,v,expiry_date,value_date,CallPut,delta):
    c_p = BlackScholes.cp(CallPut.upper())
    f = (S+fwd)
    t = BlackScholes.T(expiry_date,value_date)

    if v > 1.5:
        v = v/100
    else:
        v = v
    return f * np.exp(-c_p*norm.ppf(delta)*v*np.sqrt(t)+((v**2)/2)*t)


def GetStrikeFromDeltaPA(S,fwd,v,expiry_date,value_Date,CallPut,delta):

    c_p = BlackScholes.cp(CallPut.upper())
    f= (S+fwd)
    t = BlackScholes.T(expiry_date,value_Date)

    K_unadjust = GetStrikeFromDelta(S,fwd,v,expiry_date,value_Date,CallPut,delta)
    get_k= lambda K : abs(BlackScholes.BlackScholes(S,K,fwd,v,expiry_date,value_Date,CallPut,'BUY',1,True).delta()) - 0.25
    return optimize.newton(get_k,K_unadjust)

def Get_DN_Strike(S,fwd,v,expiry_date,value_date,deltaBase=True):

    guess_k = (S+fwd)*np.exp(-norm.ppf(0.5)*(v/100)*np.sqrt(BlackScholes.T(expiry_date,value_date))+((v/100)**2)/2)
    def opt_call(S,K,fwd,v,expiry_date,value_date,deltaBase):
        return BlackScholes.BlackScholes(S,K,fwd,v,expiry_date,value_date,'CALL','BUY',1,deltaBase).delta()
    def opt_put(S,K,fwd,v,expiry_date,value_date,deltaBase):
        return BlackScholes.BlackScholes(S,K,fwd,v,expiry_date,value_date,'PUT','BUY',1,deltaBase).delta()
    get_k = lambda K : opt_call(S,K,fwd,v,expiry_date,value_date,deltaBase) + opt_put(S,K,fwd,v,expiry_date,value_date,deltaBase)
    return optimize.newton(get_k,guess_k)

def convert_bfly(v,BF,RR):
    return BF+(-0.02*RR)+(0.0005778*abs(RR/BF))+(-0.17*RR/v)

