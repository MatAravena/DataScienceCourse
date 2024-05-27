# we'll use hard drive data as the basis for our first big-data data set. 
# pyspark enables you to access the Spark infrastructure and learn how and why you should use the pyspark.sql.DataFrame data type for large data sets.

import os
data_dir = 'HDD_logs/'
file_list = sorted(os.listdir(data_dir))
len(file_list)


# this will trhow an error for lack of memory to process all the files
import pandas as pd

# create first DataFrame
print(file_list[0])
df = pd.read_csv(data_dir + file_list[0])

# append all remaining files to DataFrame
for file in file_list[1:]:
    print(file)
    tmp_df = pd.read_csv(data_dir + file)
    df = df.append(tmp_df()




# # Big Data in PySpark
# Big data analytics processes data that achieve high values at the three Vs, i.e. data that is gathered and processed:

# in large Volumes
# in a wide Variety
# and at high Velocity




# # PySpark
from pyspark.sql import SparkSession 

# connect to Spark
spark = (SparkSession
        .builder
        .appName("Python Spark SQL HDD Analysis")
        .getOrCreate()
        )

df1 = spark.read.csv(data_dir + file_list[0] , header=True);
df1

df2 = spark.read.csv(data_dir + file_list[1] , header=True);
df2.count()

df1_2 = df1.union(df2)
df1_2.count()


df1 = spark.read.csv(data_dir + file_list[0] , header=True);
for file in file_list[1:]:
    df_temp = spark.read.csv(data_dir + file, header=True);
    df1 = df1.union(df_temp)

df1.count()

spark.stop()


# Remember:

# Big Data is big in the three Vs : Volume, Variety, and Velocity.
# pyspark gives you access to Spark, a framework to work with Big Data on computer clusters.
# If you want your DataFrame to contain more entries than your RAM can handle, use pyspark.sql.DataFrame from the pyspark.sql module.
# To use a pyspark.sql.DataFrame, you have to instantiate a SparkSession first:
#   # connect to Spark
#   spark = (SparkSession
#           .builder
#           .appName("Python Spark SQL HDD Analysis")
#           .getOrCreate()
#           )
# It is best practice to close the SparkSession with spark.stop() once you have finished your work.

