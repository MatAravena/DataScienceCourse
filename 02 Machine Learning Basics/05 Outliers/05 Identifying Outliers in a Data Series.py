# Identify improbable data points in a data series.
# Use robust measures for outlier detection.

# module import
import sqlalchemy as sa
import pandas as pd

# connection to database
engine = sa.create_engine('sqlite:///hydro.db')
connection = engine.connect()

# read out temperature data of sensor
sql_query = '''
SELECT *
FROM temperature_sensor_{}
'''
df_temp1 = pd.read_sql(sql_query.format(1), con=connection)
df_temp2 = pd.read_sql(sql_query.format(2), con=connection)
df_temp3 = pd.read_sql(sql_query.format(3), con=connection)
df_temp4 = pd.read_sql(sql_query.format(4), con=connection)
df_temp1.head()



# How can you now find out which measured values are outliers?

# How you identify an outlier depends entirely on what exactly you mean by an outlier.
# In general, an outlier is an extremely unusual data point. But how do you quantify "unusual"? How can the unusualness of a data point be summarized in a number?

# Perhaps the most widely used method of identifying outliers is based on the assumption that outliers represent very improbable data points. 
# For example, if you assume there is a normal distribution, then slightly less than 5% of the data are more than 2 standard deviations away from the mean, 


# Table that shows how likely is to find outliers depending how far are from the mean
# Distance to mean (standard deviations)	probability
# 1	                                        31.7%
# 2	                                        4.6%
# 3	                                        0.3%

# Taht's why the following code setup as outliers those datapoints with more than 3

# create empty DataFrame
df_temp4_outliers_count = df_temp4.loc[:, ['cycle_id']]

#################################################################################
# option without loop (faster)

# calculate distances to means of cycles
distance_to_mean = df_temp4.iloc[:, :-1].T - df_temp4.iloc[:, :-1].mean(axis=1)
absolute_distance_to_mean = distance_to_mean.abs()
std_distance_to_mean = absolute_distance_to_mean / df_temp4.iloc[:, :-1].std(axis=1)

# identify outliers
mask_outliers = std_distance_to_mean.T >= 3

# count outliers and add to DataFrame
df_temp4_outliers_count.loc[:, 'N_outliers'] = mask_outliers.sum(axis=1)

df_temp4_outliers_count.head()
