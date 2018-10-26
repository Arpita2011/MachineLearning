# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17 15:03:27 2018

@author: Arpita
"""

import pandas as pd

cdsSpread_data = pd.read_stata("C:\\Users\\Arpita\\Desktop\\Book\\Fall 2018\\Machine Learning\\Assignment\\cds_spread5y_2001_2016.dta")

cdsSpread_data.shape
cdsSpread_data.dtypes
cdsSpread_data.head()

qtr_Mdata = pd.read_csv("C:\\Users\\Arpita\\Downloads\\Quarterly Merged CRSP-Compustat.csv", low_memory = False)
qtr_Mdata.shape
qtr_Mdata.dtypes

qtr_Mdata.head()
cdsSpread_data['Date'] = pd.to_datetime(cdsSpread_data['mdate'])
cdsSpread_data['gvkey'] = cdsSpread_data['gvkey'].astype(float)
qtr_Mdata=qtr_Mdata.rename(columns = {'GVKEY':'gvkey'})
data = pd.merge(cdsSpread_data, qtr_Mdata, on="gvkey")







