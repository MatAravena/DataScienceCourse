# Got to know the term lazy evaluation
# Displayed a pyspark.sql.DataFrame
# Transformed a pyspark.sql.DataFrame

import pandas as pd
from pyspark.sql import SparkSession

# connect to Spark
spark = (SparkSession
        .builder
        .appName("Python Spark SQL HDD Analysis")
        .getOrCreate())


data_dir = 'HDD_logs/'
df_pandas  = pd.read_csv(data_dir + '2016-01-01.csv')
df_spark = spark.read.csv(data_dir + '2016-01-01.csv', header=True)


# lazy evaluation
# When interacting with DataFrames, Spark distinguishes between two categories: Transformations und Actions.

# Transformations are all operations that are applied to the entire data set, such as select(), groupBy(), describe(), filter(), etc.
# Actions are all operations that output data, such as count(), sum(), average(), collect(), and all operations that display data to the user.


# # Display the pyspark.sql.DataFrame
df_pandas.head()


df_spark.show(n=5)
df_spark['date', 'serial_number', 'model', 'capacity_bytes', 'failure'].show(n=5)

meta_cols = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']
df_spark_meta = df_spark[meta_cols]
df_spark_meta.show(n=5)

df_pandas_meta = df_pandas.loc[:,meta_cols]
df_pandas_meta.head()


# # Transforming a pyspark.sql.DataFrame
description_df = pd.read_csv('SMART_attributes.csv')
description_df


renamed_cols = []
for column in df_spark.columns:
    if column.find('smart_') != -1:
        ID = column.split('_')[1]
        attribute =''

        mask = description_df.loc[:,'ID'] == pd.to_numeric(ID)

        if not description_df.loc[mask,['Attribute']].empty:
            attribute = description_df.loc[mask,['Attribute']].iloc[0].values[0]
            renamed_cols.append('smart_'+attribute)

    else:
        renamed_cols.append(column)

renamed_cols



# If you put a * in front of a list and pass it to a function, all the elements of the list are treated as arguments their respective data types, separated by commas. 
# This saves you a lot of typing.

df_spark = df_spark.toDF(*renamed_cols)
df_spark.columns

df_spark['smart_read_error_rate', 'smart_spin_up_time'].show(n=5)

spark.stop()
