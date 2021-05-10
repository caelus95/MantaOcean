# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:58:14 2020

@author: psi36
"""

import math as mt


def pH2H(pH):
    H = 10**(-pH)
    return H

def pK2K(pK):
    K = 10**(-pK)
    return K

def COstar(Ct,H,K1,K2):
    COstar = (Ct*(H)**2)/((H)**2 + K1*H + K1*K2)
    return COstar


S = 35
T = 273.15+5
Ct = 1900
pH = 8.0

pK1= 3670.7/T - 62.008 + 9.7944*mt.log(T) - 0.0118*S + 0.000116*S**2

pK2= 1394.7/T + 4.777 - 0.0184*S + 0.000118*S**2

lnK0= -60.2409 + 93.4517*(100/T) + 23.3585*mt.log(100/T) + S*(0.023517 -0.023656*(T/100) + 0.0047036*(T/100)**2)

K1 = pK2K(pK1)
K2 = pK2K(pK2)

H = pH2H(pH)

COstar = COstar(Ct,H,K1,K2)

K0 = mt.exp(lnK0)

fCO2 = COstar/K0 

