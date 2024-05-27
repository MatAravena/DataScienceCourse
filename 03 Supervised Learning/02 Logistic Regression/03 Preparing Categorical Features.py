# Label encoding
# One-hot encoding


import pandas as pd
import numpy as np

df = pd.read_csv("social_media_train.csv", index_col=[0])
df.head()

features_cat = ['profile_pic', 'sim_name_username', 'extern_url', 'private']
print(features_cat)

for col_name in features_cat:
    unique_values = df.loc[:, col_name].unique()
    print("\nColumn name: {}\nUnique values: {}".format(col_name, unique_values))

# Category columns must have numerical values instead strings
df.loc[:, 'profile_pic_encoded'] = df.loc[:, 'profile_pic'].replace({'Yes': 1, 'No': 0})
df.loc[:, 'extern_url_encoded'] = df.loc[:, 'extern_url'].replace({'Yes': 1, 'No': 0})
df.loc[:, 'private_encoded'] = df.loc[:, 'private'].replace({'Yes': 1, 'No': 0})


print("Column name: {}\nUnique values: {}".format("profile_pic_encoded", df.loc[:, 'profile_pic_encoded'].unique()))
print("Column name: {}\nUnique values: {}".format("extern_url_encoded",  df.loc[:, 'extern_url_encoded'].unique()))
print("Column name: {}\nUnique values: {}".format("private_encoded",     df.loc[:, 'private_encoded'].unique()))


# One-hot encoding
# is used to represent categorical features with new binary features that only contain `0` and `1`.

# pdpipe.OneHotEncode(
#     columns=list, #column labels to be one-hot encoded
#     dummy_na=bool, #add column to indicated NaN (True=Yes, False=No)
#     exclude_columns=str or list, #name of categorial columns to be excluded from encoding
#     drop_first=bool #Whether to get k-1 dummies out of k categorical levels by removing the first level (default: True))

import pdpipe as pdp
onehot = pdp.OneHotEncode(['country'], drop_first=False)
df_example = onehot.fit_transform(df_example)


onehot = pdp.OneHotEncode(['sim_name_username'], drop_first=False)
df = onehot.fit_transform(df)
df.head()



# Remember:
# Label encoding uses my_df.replace() to create a 0/1 feature from a binary categorical feature
# One-hot encoding uses pdp.OneHotEncode() to create a set of new 0/1 features from a categorical feature with more than two categories
