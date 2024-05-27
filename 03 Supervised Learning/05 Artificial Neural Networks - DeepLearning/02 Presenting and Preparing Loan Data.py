
#define the function, the argument is a DataFrame `df` which must have the same column structure as `train`
def data_cleaner(df):
    '''This function performs all above mentioned data manipulation steps.'''

    copy=df.copy()

    # Create target column
    mask=(copy.loc[:,'int_rate'] > 14).astype('int8')
    copy.loc[:,'problem_loan']=mask

    # group 'purpose'
    mask_purpose = copy.loc[: , 'purpose'].isin(['educational', 'renewable_energy', 'house', 'wedding', 'vacation',
       'moving', 'medical'])  
    copy.loc[mask_purpose, 'purpose'] = 'other'
    
    # get only first number of zip code
    copy.loc[:,'1d_zip'] = copy.loc[:,'zip_code'].str[0]
    
    # replace term with numbers
    copy.loc[:, 'term'] = copy.loc[:, 'term'].replace({' 36 months': 36, ' 60 months': 60})
    
    # extract numeric values of emp_length into emp_length_num
    copy.loc[:, 'emp_length'] = copy.loc[:,'emp_length'].str.replace(pat=r'< 1', repl='0')
    copy.loc[:,'emp_length_num'] = copy.loc[:,'emp_length'].str.extract(r'(\d+)', expand=False)
    
    # create new column unemployed
    copy.loc[:, 'unemployed'] = 0  # fill with 0 (person is employed)
    mask_unemployed = (copy.loc[:, 'emp_length'].isna()) & (copy.loc[:, 'emp_title'].isna())  # select rows with missing job title and employment length
    copy.loc[mask_unemployed, 'unemployed'] = 1  # fill selected rows with 0 (person is not employed)

    # delete rows with employment length but missing job title
    mask_na = (copy.loc[:, 'emp_title'].isna()) & (~copy.loc[:, 'emp_length'].isna())
    copy = copy.drop(copy.index[mask_na])

    # delete rows with job title but missing employment length
    mask_na = (~copy.loc[:, 'emp_title'].isna()) & (copy.loc[:, 'emp_length'].isna())
    copy = copy.drop(copy.index[mask_na])

    # fill missing values in emp_length_num with -1 to prevent dropping them
    copy.loc[:, 'emp_length_num'] = copy.loc[:, 'emp_length_num'].fillna(-1)
    copy.loc[:, 'emp_length_num'] = pd.to_numeric(copy.loc[:, 'emp_length_num'])
    
    # drop irrelevant columns. We have not introduced `year` and `month`, thus we do not need to drop them
    copy = copy.drop(['id', 'issue_d', 'emp_title', 'emp_length', 'title', 'zip_code', 'int_rate'], axis=1)
    
    # drop missing values
    
    copy = copy.dropna()
    
    return copy   

clean_loans_train = data_cleaner(loans_train)
clean_loans_val = data_cleaner(loans_val)


# import relevant transformers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# define columns for OHE
columns=['grade', 'sub_grade', 'home_ownership', 'verification_status','1d_zip', 'purpose']

# set up pipeline
ohe = OneHotEncoder(sparse=False)
encoder = ColumnTransformer([('OHE',ohe, columns)], remainder='passthrough')

# fit encoder
encoder.fit(clean_loans_train)

# restore column names for final DataFrames
ohe_names = encoder.named_transformers_['OHE'].get_feature_names(columns)
remaining_names = encoder._df_columns[encoder._remainder[2]]

#apply encoding and create DataFrames
final_loans_train = pd.DataFrame(encoder.transform(clean_loans_train), columns = list(ohe_names) + list(remaining_names))
final_loans_val = pd.DataFrame(encoder.transform(clean_loans_val), columns = list(ohe_names) + list(remaining_names))

#check shapes
print(final_loans_train.shape)
print(final_loans_val.shape)


# split train and validation data into features and target
target_train = final_loans_train.loc[:, 'problem_loan']
features_train = final_loans_train.drop('problem_loan', axis=1)
target_val = final_loans_val.loc[:, 'problem_loan']
features_val = final_loans_val.drop('problem_loan', axis=1)

# save data as pickle
features_train.to_pickle('features_train.p')
target_train.to_pickle('target_train.p')
features_val.to_pickle('features_val.p')
target_val.to_pickle('target_val.p')