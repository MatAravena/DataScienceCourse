# Registered a pyspark.sql.DataFrame as an SQL table
# Aggregated the data in your pyspark.sql.DataFrame with SQL queries
# applied a Python function to the values within the pyspark.sql.DataFrame



from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os

# locate Data
data_dir = "HDD_logs/"

# connect to Spark
spark = (SparkSession.builder.appName("Python Spark SQL HDD Analysis").getOrCreate())

file_list = sorted(os.listdir(data_dir))

df = spark.read.csv(data_dir + file_list[0] , header=True);

for file in file_list[1:3]:
    df_temp = spark.read.csv(data_dir + file, header=True);
    df = df.union(df_temp)


metacols = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure']


# describe
df[metacols].describe().show()


failure_describe_values = df[metacols].describe().select("failure").collect()
failure_describe_values



# # SQL queries with Spark

df_meta = df[metacols]
df_meta.registerTempTable('meta_data')

query = """SELECT DISTINCT(model) AS model_name
            FROM meta_data
            ORDER BY model_name"""

model_names = spark.sql(query)
model_names.show()


#define rules for tagging the model names
def clean_name(name):
    """Extract HDD Brand from model_name.
    Args:
        name (str) : Model name of HDD
    Returns:
        str : Brand Tag of HDD
    """
    if name[:2] == "ST":  # seagate has no space in model number
        brand = "Seagate"
    else:
        brand = name.split(" ")[0]

    # check if brand was saved
    if brand:
        return brand
    else:
        print("Could not extract Brand from: {}".format(name))
        return "Other"


# Convert the method to let spark sql reads it
# user defined function format, or udf
clean_name_udf = udf(clean_name)


# Pro!
df_meta = df_meta.withColumn(colName='brand', col=clean_name_udf('model'))
df_meta.show()

# Register this table in the sql table thing
df.registerTempTable('meta_data')


# # Which hard disk models are responsible for the failures?
my_query = """SELECT model, SUM(failure) AS fails, COUNT(DISTINCT(serial_number)) as no_model FROM meta_data group by model """
spark.sql(my_query).show()



my_query = """
            select * , fails/no_model as failure_rate
            from
                    (SELECT model,
                    SUM(failure) AS fails,
                    COUNT(DISTINCT(serial_number)) as no_model
                    FROM meta_data group by model)
            ORDER BY failure_rate DESC
            """

spark.sql(my_query).show()






# Remember:

# The my_spark_df.select() method always returns a DataFrame even if you only select a single column.
# To apply custom functions to the pyspark.sql.DataFrame, you have to convert them to an udf object with pyspark.sql.functions.udf().
# You can register a pyspark.sql.DataFrame as a table in Spark with my_spark_df.registerTempTable('my_TableName') and then apply SQL queries to it with my_SparkSession.sql().