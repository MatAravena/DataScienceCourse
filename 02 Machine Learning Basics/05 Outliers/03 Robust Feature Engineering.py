# What we'll learn
# What robust procedures are.
# How to calculate the median absolute deviation as an alternative to standard deviation.


# module import
import sqlalchemy as sa
import pandas as pd

# connection to database
engine = sa.create_engine('sqlite:///hydro.db')
connection = engine.connect()

# read out temperature data from sensors
sql_query = '''
SELECT *
FROM temperature_sensor_{}
'''
df_temp1 = pd.read_sql(sql_query.format(1), con=connection)
df_temp2 = pd.read_sql(sql_query.format(2), con=connection)
df_temp3 = pd.read_sql(sql_query.format(3), con=connection)
df_temp4 = pd.read_sql(sql_query.format(4), con=connection)

df_temp1.head()

# The 60 seconds per cycle represent the features. However, it is a fallacy to assume that since all features describe the temperature, 
# our real feature is the temperature. This is a typical property of a time series.

# Since we are now looking at the cooling capacity independently of the time and we want to predict it from the temperature data, 
# the time component is of very little use to us. We want to break down the temporal developments into characteristic numbers and then use these as independent temperature features

df = df_temp1.loc[:,['cycle_id']] 

# Calculate the average temperature values per cycle for the first temperature sensor

# axis parameter to 0 Column
# axis parameter to 1 rows

# temperature sensor 1 (mean and standard deviation)
df.loc[:, 'temp1_central'] = df_temp1.mean(axis=1)
df.loc[:,'temp1_dispersion'] = df_temp1.std(axis=1)

# temperature sensor 2 (mean and standard deviation)
df.loc[:, 'temp2_central'] = df_temp2.mean(axis=1)
df.loc[:, 'temp2_dispersion'] = df_temp2.std(axis=1)

# temperature sensor 3 (mean and standard deviation)
df.loc[:, 'temp3_central'] = df_temp3.mean(axis=1)
df.loc[:, 'temp3_dispersion'] = df_temp3.std(axis=1)

# temperature sensor 4 (mean and standard deviation)
df.loc[:, 'temp4_central'] = df_temp4.mean(axis=1)
df.loc[:, 'temp4_dispersion'] = df_temp4.std(axis=1)


# You should also swap around rows and columns with df.T so that one line is plotted for each cycle.
import matplotlib.pyplot as plt

#drawing figure
fig, ax = plt.subplots()

# plot line chart
df_temp4.iloc[[149, 150], :-1].T.plot(legend=True, ax=ax)

# optimise line chart
ax.set(xlabel='Time since cycle start [sec]',
       ylabel='Temperature [°C]',
       title='Temperature sensor 4 data of hydraulic pump')
ax.set_xticklabels(range(-10, 71, 10))


df_temp4.iloc[[149, 150], :].mean(axis=1)
# 149    41.360111
# 150    41.152167
# dtype: float64

df_temp4.iloc[[149, 150], :].std(axis=1)
# 149    1.915462
# 150    0.053144
# dtype: float64



# Now if you wanted to train a machine learning model based on the mean and the standard deviation, the algorithm would be strongly influenced by the outliers, 
# since it relies exclusively on numerical differences and would learn the outlier's influence. 
# So what can you do to protect a machine learning model from this kind of misleading information? Generally speaking, there are three approaches:

# Delete outliers (either delete the data point or just mark the value as NaN).
# Replace outlier values (with a consistent extreme value that is still "acceptable", or with data imputation - using a probable value).
# Minimize the influence of outliers (transform features or use robust methods).



# The median is the typical value of a distribution. It marks the middle value if you put the data in order from the smallest to the largest value. 
# Note that for the calculation of the median it does not matter whether extreme values occur at the upper or lower end of the value distribution.

df_temp4.iloc[[149, 150], :].median(axis=1)

# If you have outliers in your data, it is therefore advisable to take the median rather than the arithmetic mean as the central value of a data series.

# median absolute deviation (MAD)
# This describes the median of the distances from all data points to the median value of the data series. 


# Important: The my_df.mad() method from pandas does not calculate the median absolute deviation, but the mean absolute deviation, 
# Example: the mean value of the deviations from the median value of the data. 
# This also reduces the influence of outliers, but not that much. We recommend using the median absolute deviation instead 
# if you want to keep the influence of outliers as small as possible. So you have to use the mad() function from the statsmodels.robust module!

from statsmodels.robust import mad
mad(df_temp4.iloc[[149, 150], :-1],axis=1)

# array([0.01186082, 0.07561271])

df_robust = df_temp1.loc[:,['cycle_id']]

for sensor in range(1, 5):  # for each thermometer
    sql_query = '''
    SELECT *
    FROM temperature_sensor_{}
    '''.format(sensor)
    df_tmp = pd.read_sql(sql_query, con=connection)

    df_robust.loc[:, 'temp{}_central'.format(sensor)] = df_tmp.median(axis=1)
    df_robust.loc[:, 'temp{}_dispersion'.format(sensor)] = mad(df_tmp.iloc[:, :-1], axis=1)
df_robust.head()


# **Congratulations:** You've got to know robust alternatives to the arithmetic mean and standard deviation. The median and *median absolute deviation* are barely influenced by outliers. 
# This is the kind of behavior you normally want. Next we'll look at how similar the robust and non-robust key figures in this data set are.


# Comparing robust and non-robust values
# Draw a scatter plot of the mean values (x-axis) and median values(y-axis) of the fourth temperature sensor of all the cycles.
import seaborn  as sns
sns.scatterplot(x=df.loc[:, 'temp4_central'], y=df_robust.loc[:, 'temp4_central']);
# Visualize the standard deviation (x-axis) and the *median absolute deviation* (y-axis) of the fourth temperature sensor of all the cycles in a scatter plot.
sns.scatterplot(x=df.loc[:, 'temp4_dispersion'], y=df_robust.loc[:, 'temp4_dispersion']);

# As you can see just by looking at it, the values here do not lie on a variable like the mean and median. 
# From the plot it can be concluded that standard deviation and median absolute deviation are generally very dissimilar and only the latter is robust against outliers.



# table names in database
tables = ['cooling_efficiency',
          'cooling_power',
          'machine_efficiency',
          'temperature_sensor_1',
          'temperature_sensor_2',
          'temperature_sensor_3',
          'temperature_sensor_4',
          'volume_flow_sensor_1',
          'volume_flow_sensor_2']

# columns names in DataFrames
col_names = ['cool_eff',
             'cool_power',
             'mach_eff',
             'temp_1',
             'temp_2',
             'temp_3',
             'temp_4',
             'flow_1',
             'flow_2']

# initialise DataFrames
df = df_temp1.loc[:, ['cycle_id']]
df_robust = df_temp1.loc[:, ['cycle_id']]

for s in range(len(tables)):  # for each sensor
    
    # read in data from database
    sql_query = '''
    SELECT *
    FROM {}
    '''.format(tables[s])
    df_tmp = pd.read_sql(sql_query, con=connection)
    
    # non-robust summary values
    df.loc[:, '{}_central'.format(col_names[s])] = df_tmp.mean(axis=1)
    df.loc[:, '{}_dispersion'.format(col_names[s])] = df_tmp.std(axis=1)
    
    # robust summary values
    df_robust.loc[:, '{}_central'.format(col_names[s])] = df_tmp.median(axis=1)
    df_robust.loc[:, '{}_dispersion'.format(col_names[s])] = mad(df_tmp.iloc[:, :-1], axis=1)

# pickle data
df.to_pickle('hydro_data.p')
df_robust.to_pickle('robust_hydro_data.p')

# close connection to data base
connection.close()