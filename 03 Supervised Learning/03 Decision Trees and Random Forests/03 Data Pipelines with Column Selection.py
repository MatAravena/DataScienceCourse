# module import
import pandas as pd

# data gathering
df_train = pd.read_csv('attrition_train.csv')
df_test = pd.read_csv('attrition_test.csv')

#extract features and target
features_train = df_train.drop('attrition', axis=1)
target_train = df_train.loc[:,'attrition']

features_test = df_test.drop('attrition', axis=1)
target_test = df_test.loc[:,'attrition']

# look at raw data
df_train.head()

col_correlated = ['totalworkingyears',
                  'years_atcompany',
                  'years_currentrole',
                  'years_lastpromotion',
                  'years_withmanager']



# Create a pipeline that handles features differently
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

std_pca = Pipeline([('std', StandardScaler()),
                    ('pca', PCA(n_components=0.8))])


# to apply our PCA to col_correlated
from sklearn.compose import ColumnTransformer

# ColumnTransformer(transformers=list, # List of Tuples like ('step_name',transformer,[columns to apply transformer to])
#                   remainder=str, # Strategy of what to do with remainder;
#                                  # - 'drop': delete,
#                                  # - 'passthrough': append to output,
#                                  # - transformer: pipe to another transformer and append result
#                   n_jobs=int)    # number of CPU cores to use


colTransformer = ColumnTransformer([('pca', std_pca, col_correlated)])

colTransformer.fit_transform(features_train)
df_ColTrans = colTransformer.transform(features_test)
df_ColTrans.shape

keep_cols = ['age',
             'gender',
             'businesstravel',
             'distancefromhome',
             'education',
             'joblevel',
             'maritalstatus',
             'monthlyincome',
             'numcompaniesworked',
             'overtime',
             'percentsalaryhike',
             'stockoptionlevels',
             'trainingtimeslastyear']

col_dropper = ColumnTransformer([('drop_unused_cols', 'passthrough', keep_cols)])

# Check if all the values are the same
col_dropper.fit_transform(features_train)
(features_test[keep_cols].values == col_dropper.transform(features_test)).all()



# Create transformer with a remainder 
# Basicamente crea nuevas columnas transformadas especificadas como columnas correlacionadas y el resto lo anade al valor de salida sin transformarlas
corr_transformer = ColumnTransformer([('pipe_std_pca_corrcols', std_pca, col_correlated)], remainder=col_dropper)

print("Manual:  (1029, 15)")
piped_out_arr = corr_transformer.fit_transform(features_train)
print("Piped : ", piped_out_arr.shape)
print("Piped : ", features_train.shape)

col_names = ['pca_years_0','pca_years_1'] + keep_cols
output = pd.DataFrame(piped_out_arr, columns=col_names)
output.head()



# So to summarize

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

col_correlated = ['totalworkingyears',
                  'years_atcompany',
                  'years_currentrole',
                  'years_lastpromotion',
                  'years_withmanager']

keep_cols = ['age',
             'gender',
             'businesstravel',
             'distancefromhome',
             'education',
             'joblevel',
             'maritalstatus',
             'monthlyincome',
             'numcompaniesworked',
             'overtime',
             'percentsalaryhike',
             'stockoptionlevels',
             'trainingtimeslastyear']

std_pca = Pipeline([('std', StandardScaler()), 
                    ('pca', PCA(n_components=0.8))])
        
col_dropper = ColumnTransformer([('drop_unused_cols', 'passthrough', keep_cols)],
                                remainder='drop')

corr_transformer = ColumnTransformer([('pipe_std_pca_corrcols', std_pca, col_correlated)],
                                     remainder=col_dropper)

col_names = ['pca_years_0','pca_years_1'] + keep_cols

#load data
df_train = pd.read_csv('attrition_train.csv')
df_test = pd.read_csv('attrition_test.csv')

#split into features
features_train = df_train.drop('attrition', axis=1)
features_test = df_test.drop('attrition', axis=1)

#clean data
corr_transformer.fit_transform(features_train)
pd.DataFrame(corr_transformer.transform(features_test), columns=col_names)



import pickle
pickle.dump(corr_transformer, open("pipeline.p","wb"))
pickle.dump(col_names, open("col_names.p","wb"))


# Remember:
# With ColumnTransformer you can apply transformer to selected columns.
# If you use the string 'passthrough' instead of a transformer when defining an sklearn pipeline, you can output data exactly as it was output.
