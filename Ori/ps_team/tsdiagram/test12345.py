# -*- coding: utf-8 -*-
#""""
#Created on Tue Feb 11 17:09:59 2020
#
#@author: user
#""""

from openpyxl import Workbook
from openpyxl import load_workbook
import numpy as np

# WDtest = load_workbook(filename='./C:/Users/user/Desktop/소희/논문/해무 자료/바람장미/발생시/외연도/외연도-3월.xlsx')

from os.path import join
import pandas as pd
test = pd.read_excel('D:/temp_task/외연도-3월.xlsx' , header = None)
test2 = open('D:/temp_task/20190703_01.cnv')
test2.close

test3 = pd.read_csv('D:/temp_task/20190703_01.cnv',sep=' ',delim_whitespace='Ture')
test3 = pd.read_csv('D:/temp_task/20190703_01.cnv')

test4 = pd.read_table('D:/temp_task/20190703_01.cnv',sep=' ')


test.columns = ['year','month','day','hour','WS','WD']
test["Freq"] = ''


 #np.where((ELat >= llcrnrlat) & (ELat <= urcrnrlat))

WD = test["WD"]
def direc(WD):
    
    if   (WD >=11.25) & (WD <= 33.74):
        D = "북북동"
    elif 33.75 <= WD <= 56.24:
        D = "북동"
    elif 56.25 <= WD <= 78.74 :
        D="동북동"
    elif 78.75 <= WD <= 101.24 :
        D="동"
    elif 101.25 <= WD <= 123.74 :
        D="동남동"
    elif 123.75 <= WD <= 146.24 :
        D="남동"
    elif 146.25 <= WD <= 168.74 :
        D="남남동"
    elif 168.75 <= WD <= 191.24 :
        D="남"
    elif 191.25 <= WD <= 213.74 :
        D="남남서"
    elif 213.75 <= WD <= 236.24 :
        D="남서"
    elif 236.25 <= WD <= 258.74 :
        D="서남서"
    elif 258.75 <= WD <= 281.24 :
        D="서"
    elif 281.25 <= WD <= 303.74 :
        D="서북서"
    elif 303.75 <= WD <= 326.24 :
        D="북서"
    elif 326.25 <= WD <= 348.74 :
        D="북북서"
    else:
        D= "북"
    return D
          
n=0
for i in test.WD:
    d = direc(i)
    test.year[n] = d
    n+=1
        
        
        
        
        
        
        
        
        
        