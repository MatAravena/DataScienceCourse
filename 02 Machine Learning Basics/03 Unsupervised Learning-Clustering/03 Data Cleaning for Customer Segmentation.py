import pandas as pd
df = pd.read_csv('online_retail_data.csv') 
df.head()

df_cancel = pd.read_csv('online_retail_cancellations_data.csv')
df_cancel.shape

df.info() 
df.isna().sum()
df = df.dropna(subset=['CustomerID'])
df.isna().sum()

df.loc[:, 'StockCode'] = df.loc[:, 'StockCode'].astype('category')
df.loc[:, 'Country'] = df.loc[:, 'Country'].astype('category')
df.loc[:, 'InvoiceDate'] = pd.to_datetime(df.loc[:, 'InvoiceDate'])
df.loc[:, 'Description'] = df.loc[:, 'Description'].astype(str)

df_cancel.loc[:, 'StockCode'] = df_cancel.loc[:, 'StockCode'].astype('category')
df_cancel.loc[:, 'Country'] = df_cancel.loc[:, 'Country'].astype('category')
df_cancel.loc[:, 'InvoiceDate'] = pd.to_datetime(df_cancel.loc[:, 'InvoiceDate'])
df_cancel.loc[:, 'Description'] = df_cancel.loc[:, 'Description'].astype(str)


# Preparing data
df.loc[:,'Revenue'] = df.loc[:,'Quantity'] * df.loc[:,'UnitPrice']
df_cancel.loc[:,'Revenue'] = df_cancel.loc[:,'Quantity'] * df_cancel.loc[:,'UnitPrice']

groups = df.groupby('CustomerID')

df_customers = groups.agg({'InvoiceNo': 'nunique', 'Revenue': 'sum', 'Quantity':'sum' })
df_customers.head()

groups_cancel  = df_cancel.groupby('CustomerID')
df_cancel_customers = groups_cancel.agg({'InvoiceNo': 'nunique', 'Revenue': 'sum', 'Quantity':'sum' })
df_cancel_customers.head()

add_revenues = df_customers.copy().loc[:, 'Revenue'] + df_cancel_customers.copy().loc[:, 'Revenue']
add_revenues.head()

df_customers.loc[:, 'Revenue'] =  df_customers.loc[:, 'Revenue'].add(df_cancel_customers.loc[:, 'Revenue'], fill_value=0)
df_customers.loc[:, 'Quantity'] = df_customers.loc[:, 'Quantity'].add(df_cancel_customers.loc[:, 'Quantity'], fill_value=0)
df_customers.head()


# Checking values

mask = df_customers.loc[:, 'Revenue'] <0
df_customers.loc[mask, :]

#mask = df_customers.loc[:, 'Revenue'] <0
mask = df_customers.loc[:, 'Revenue'] > 0
df_customers = df_customers.loc[mask, :]

mask = df_customers.loc[:, 'Quantity'] > 0
df_customers = df_customers.loc[mask, :]
df_customers

mask = df_customers.loc[:, 'Revenue'] <= 0
df_customers.loc[mask, :]

RevenueMean = df_customers.loc[:, 'Revenue'] / df_customers.loc[:, 'InvoiceNo']
QuantityMean = df_customers.loc[:, 'Quantity'] / df_customers.loc[:, 'InvoiceNo']
PriceMean = df_customers.loc[:, 'Revenue'] / df_customers.loc[:, 'Quantity']
df_customers.info()
print('RevenueMean', RevenueMean.head())
print('QuantityMean', QuantityMean.head())
print('PriceMean', PriceMean.head())

df_customers.loc[:, 'RevenueMean'] = RevenueMean
df_customers.loc[:, 'QuantityMean'] = QuantityMean
df_customers.loc[:, 'PriceMean'] = PriceMean



df_customers.loc[:, 'Quantity'] = df_customers.loc[:, 'Quantity'].astype('int')

df_customers.info()

df_customers.to_pickle('customer_data.p')



# Remember:

# Cleaning and preparing data is a very important part of a data scientist's work
# Add two pandas objects that contain NaN values together with my_Series_or_DataFrame.add()
# Aggregate data with my_df.groupby().agg().
# Store DataFrames as a pickle by using my_df.to_pickle('filename')