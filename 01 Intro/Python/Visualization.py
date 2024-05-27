import pandas as pd
df_publisher_sum = pd.read_csv('publisher_sum.csv')
df_publisher_sum

df_publisher_sum = pd.read_csv('publisher_sum.csv',index_col='publisher')

import matplotlib.pyplot as plt
plt.style.available

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=[11, 8])
ax.ticklabel_format(style="plain")


# We can create a visualization with the method my_df.plot(). Its kind parameter determines what kind of graph it creates. You can assign the following str options to kind:

# 'line' : Line graph (default)
# 'bar' : Column chart (vertical bar chart)
# 'barh' : Bar chart (horizontal)
# 'hist' : Histogram
# 'box' : Box plots
# 'kde' : Density plot (kernel density estimation)
# 'density' : The same as 'kde'
# 'area' : area chart
# 'pie' : Pie chart
# 'scatter' : Scatter plot
# 'hexbin' : Hexbin plot
# What kind of graph is best for our data? Since we grouped the data by category (publishers), a column chart would make sense.



fig, ax = plt.subplots(figsize=[11, 8])
ax.ticklabel_format(style="plain")
df_publisher_sum.sort_values(by='daily_gross_sales', ascending=False).plot(kind='barh', ax=ax)
fig



# Adjusting visualitzation

fig, ax = plt.subplots(figsize=[11, 8])
ax.ticklabel_format(style="plain")

df_publisher_sum.loc[:, 'daily_gross_sales'].sort_values(ascending=False).plot(kind='barh', ax=ax)
ax.set(title='Daily Gross sales',xlabel='value', ylabel='Publishers' )
fig

fig.savefig('myfig.png')


