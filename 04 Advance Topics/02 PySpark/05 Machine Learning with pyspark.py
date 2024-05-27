# Implemented one-hot encoding with pyspark
# Performed logistic regression with pyspark
# Evaluated your predictions with pyspark.



from pyspark.sql import SparkSession

#connect to Spark
spark = (SparkSession
         .builder
         .appName("ML mit SparkML")
         .getOrCreate()
        )

df = spark.read.csv('aggregated_HDD_Data.csv', header=True, inferSchema=True)

df.printSchema()


# Change string categories to numeric values
from pyspark.ml.feature import StringIndexer

brand_indexer = StringIndexer(inputCol="brand", outputCol="brand_indexed")  # initialize indexer
brand_indexer = brand_indexer.fit(df)  # fit indexer to dataframe
df = brand_indexer.transform(df)  # encode brand

model_indexer = StringIndexer(inputCol="model", outputCol="model_indexed")  # initialize indexer
model_indexer = model_indexer.fit(df)  # fit indexer to dataframe
df = model_indexer.transform(df)  # encode brand



# Drop previous columns
df['model', 'model_indexed','brand_indexed','brand'].show()
df = df.drop('brand','model')



# One-hot encode
# OneHotEncoder(inputCols=list,   # list with names of categorical columns
#               outputCols=list)  # list with names of new columns

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols=['brand_indexed', 'model_indexed'],   # list with names of categorical columns
              outputCols=['brand_onehot', 'model_onehot'])  # list with names of new columns
encoder = encoder.fit(df)
df = encoder.transform(df)

df = df.drop('brand_indexed', 'model_indexed')


# rename the target column
# rename target col to label -> spark default for target
df = df.withColumnRenamed("failure", "label")


# In the following code cell, we first define the columns that 'features' should contain.
# Then we use VectorAssembler to merge these columns into one. VectorAssembler doesn't have the my_transformer.fit() method, only my_transformer.transform().

# select all columns that we want to use as features
feature_cols = [col for col in df.columns if col not in ['serial_number', 'days_live', 'label']]

# import and initialize VectorAssembler
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_cols,
                            outputCol="features")

# Now let us use the transform method to transform our dataset
df = assembler.transform(df)

# Test datasets
#from sklearn.model_selection import train_test_split
df_train, df_test = df.randomSplit([0.9, 0.1])




# # Logistic regression with pyspark

df_train.createOrReplaceTempView('train_set')

my_query = """
            SELECT COUNT(label) as count , label
            FROM train_set
            Group by label
            """
spark.sql(my_query).show()

df_train_classes = spark.sql(my_query).toPandas()

df_train_count = df_train.count()

#   297|    1|
# 29674|    0|

df_train_classes.index = df_train_classes.loc[:, 'label']
weights = df_train_count / df_train_classes.loc[:, 'count']

# Adding the weights to the main DF
from pyspark.sql.functions import when
df_train = df_train.withColumn("weights",
              when(df_train["label"] == 0, weights.loc[0]).otherwise(weights.loc[1]))


# Creating the LogisticRegression model
from pyspark.ml.classification import LogisticRegression
model  = LogisticRegression(weightCol='weights')

#Fitting the model
model = model.fit(df_train)

#Predictions in pyspark
df_test_pred = model.transform(df_test)

# df_test_pred is a DataFrame again, which contains the 'prediction' column as well as the features:
df_test_pred.select('prediction').show(10)


#evaluate
pred_summary = model.evaluate(df_test)


print(pred_summary.accuracy)
print(pred_summary.recallByLabel)
print(pred_summary.areaUnderROC)

# 0.9410889616185659
# [0.9413357400722022, 0.918918918918919]
# 0.9614555891631729

spark.stop()



# Remember:

# You will find everything you need for machine learning with Spark under pyspark.ml
# You can implement label encoding and one-hot encoding with StringIndexer and OneHotEncoder respectively
# Convert your features to a column of vectors with VectorAssembler(inputCols, outputCol)
# You fit a model in pyspark with my_model.fit() and generate predictions with my_model.transform()
# Evaluate your model for a data set with my_model.evaluate()