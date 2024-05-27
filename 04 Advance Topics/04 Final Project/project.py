# import modules
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import RANSACRegressor
from statsmodels.robust import mad


readableCols = ['IsBadBuy', 'PurchDate', 'Auction', 'VehYear', 'VehicleAge', 'Make',
       'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 'WheelTypeID',
       'WheelType', 'VehMileage', 'Nationality', 'Size', 'AmericanTop',
       'AcquisitionAuctionAvgPrice', 'AcquisitionAuctionCleanPrice',
       'AcquisitionRetailAvgPrice', 'AcquisitonRetailCleanPrice',
       'CurrentAuctionAvgPrice', 'CurrentAuctionCleanPrice',
       'CurrentRetailAvgPrice', 'CurrentRetailCleanPrice',
       'PRIMEUNIT', 'GuartAtAuction', 'BuyerIdAtPurch', 'ZIPAtPurch', 'StateAtPurch', 'VehBCost',
       'IsOnlineSale', 'WarrantyCost']

categorical_columns = ['IsBadBuy', 'Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 
                       'WheelTypeID', 'WheelType', 'Nationality', 'Size', 'AmericanTop', 'PRIMEUNIT', 
                       'GuartAtAuction', 'BuyerIdAtPurch', 'ZIPAtPurch', 'StateAtPurch', 'IsOnlineSale']

continuous_columns = ['IsBadBuy','VehYear', 'VehicleAge', 'VehMileage', 
                       'AcquisitionAuctionAvgPrice', 'AcquisitionAuctionCleanPrice',
                       'AcquisitionRetailAvgPrice', 'AcquisitonRetailCleanPrice',
                       'CurrentAuctionAvgPrice', 'CurrentAuctionCleanPrice',
                       'CurrentRetailAvgPrice', 'CurrentRetailCleanPrice',
                      'VehBCost', 'WarrantyCost']

correlated_cols_Acq = ['AcquisitionAuctionAvgPrice', 'AcquisitionAuctionCleanPrice',
                       'AcquisitionRetailAvgPrice', 'AcquisitonRetailCleanPrice']

correlated_cols_Current = ['CurrentAuctionAvgPrice', 'CurrentAuctionCleanPrice',
                           'CurrentRetailAvgPrice', 'CurrentRetailCleanPrice']

df = pd.read_csv("data_train.csv")
df.columns = readableCols

def cleaning_data(df):
    df = df.applymap(lambda s: s.upper() if type(s) == str else s)

    # PRIMEUNIT and GuartAtAuction
    df.loc[:, 'PRIMEUNIT'].fillna('NO', inplace=True)
    df.loc[:, 'GuartAtAuction'].fillna('YELLOW', inplace=True)

    dict_GuartAtAuction = {'GREEN': 2, 'YELLOW': 1,  'RED': 0}
    df.loc[:, 'GuartAtAuction'] = df.loc[:, 'GuartAtAuction'].replace(dict_GuartAtAuction)

    dict_PRIMEUNIT = {'YES': 1, 'NO': 0}
    df.loc[:, 'PRIMEUNIT'] = df.loc[:, 'PRIMEUNIT'].replace(dict_PRIMEUNIT)



    df = df.dropna(subset=['Trim','SubModel','Color','Transmission','WheelTypeID','WheelType','Nationality'
                      ,'Nationality','Size','VehBCost'])
    
    impute_knn= KNNImputer(n_neighbors=5)
    col_num = correlated_cols_Acq + correlated_cols_Current

    imputed_data= impute_knn.fit_transform(df[col_num])

    imputed_df= pd.DataFrame(imputed_data, columns=col_num, index=df.index)

    df= pd.concat([df.drop(columns=col_num), imputed_df], axis=1)



    for col in df[categorical_columns]:
        df.loc[:,col] = df.loc[:,col].astype('category')

    df.loc[:, 'PurchDate'] = pd.to_datetime(df.loc[:, 'PurchDate'], unit='s')



startOver()
splitDataSets(df)
