import pandas as pd
df_customers = pd.read_pickle('customer_data.p')
df_customers.head()

df = pd.read_csv('online_retail_data.csv', parse_dates = ['InvoiceDate'])
df = df.dropna()
df.loc[:, 'CustomerID'] = df.loc[:, 'CustomerID'].astype(int)

groups = df.groupby('CustomerID')
dictionary = {'InvoiceDate': ['first', 'last']}
df_customers_date = df.groupby('CustomerID').agg(dictionary)

df_customers_date.head()



df_customers_date.loc[:, 'InvoiceDate']

df_customers_date.loc[:, 'DaysBetweenInvoices'] = df_customers_date.loc[:, 'InvoiceDate']['last'] - df_customers_date.loc[:, 'InvoiceDate']['first']
df_customers_date.loc[:, 'DaysBetweenInvoices'] = df_customers_date.loc[:, 'DaysBetweenInvoices'].div( df_customers.loc[:, 'InvoiceNo'], fill_value=1)
df_customers_date.head()


df_customers_date.loc[:, 'DaysBetweenInvoices'] = df_customers_date.loc[:, 'DaysBetweenInvoices'].dt.days

date_now = pd.to_datetime('2012-01-01 00:00:00')

df_customers_date.loc[:, 'DaysSinceInvoice'] = date_now - df_customers_date.loc[:, ('InvoiceDate', 'last')]
df_customers_date.head()

df_customers_date.loc[:, 'DaysSinceInvoice'] = df_customers_date.loc[:, 'DaysSinceInvoice'].dt.days

df_customers =pd.merge(df_customers, 
                       df_customers_date.loc[:, ['DaysBetweenInvoices', 'DaysSinceInvoice']], 
                       left_index=True, 
                       right_index=True, 
                       how='left')

df_customers.columns =[
    'InvoiceNo',
    'Revenue',
    'Quantity',
    'RevenueMean',
    'QuantityMean',
    'PriceMean',
    'DaysBetweenInvoices',
    'DaysSinceInvoice'
]

df_customers.to_pickle('customer_data_prepared.p')


# Remember:

# Import a DataFrame from a pickle with pd.read_pickle()
# Use a tuple to select hierarchical columns in a DataFrame
# Divide DataFrames with different lengths with my_df.div()
# Merge DataFrames with df = pd.merge(df_left, df_right, left_index=True, right_index=True, how='left')

