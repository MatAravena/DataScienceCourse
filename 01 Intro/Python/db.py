import os
os.listdir()

import sqlalchemy as sa
import pandas as pd

engine = sa.create_engine('sqlite:///earnings.db')
connection = engine.connect()

df = pd.read_sql('SELECT * FROM earnings', connection)
df.head()
df = pd.read_sql('SELECT Publisher, daily_units_sold, daily_gross_sales, daily_bestbooks_revenue, daily_publisher_revenue FROM earnings', connection)
df.head()

print(df.loc[:,'publisher'].unique())
len(df.loc[:,'publisher'].unique())

df.isna().sum()
df = df.dropna()
df.describe()

df_publisher_sum = df.groupby('publisher').sum()
df_publisher_sum

df_publisher_sum.sort_values(by='daily_units_sold', ascending=False)


# Remember:

# Create a connection to a database with sqlalchemy.create_engine() and my_engine.connect().
# Summarize the most important values of a DataFrame with my_DataFrame.describe().
# Use my_DataFrame.to_csv() to save a DataFrame as a .csv file.
