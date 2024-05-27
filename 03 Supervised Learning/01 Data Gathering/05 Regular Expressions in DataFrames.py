# How to extract data from strings in a DataFrame with regular expressions
# How to use regular expressions to replace sections of text

# Extracting and replacing text data from DataFrames
import pandas as pd
df_company = pd.read_pickle('company_data.p')
df_company.head()

df_company = df_company.loc[:, ['Operating income', 'Net income', 'Revenue', 'Total assets', 'Total equity']]
df_company.head()


# Important: my_series.str.extract() returns a DataFrame by default. 
# It contains one column for each capture group in the search pattern. 
# As a result, the column you get back can't be added to without using additional functions. So use the parameter expand=False. This ensures that the output is a Series.

df_company.loc[:, 'Total assets year'] = df_company.loc[:, 'Total assets'].str.extract(r'(\d{4})', expand=False)
df_company.loc[:, 'Total assets year']

df_company.loc[:, 'Total assets value'] = df_company.loc[:, 'Total assets'].str.extract(r'([\d.,]+\s?[a-zA-Z]+)', expand=False)
df_company.loc[:, 'Total assets value']
# Adidas          15.612 billion

# The words in our case are 'million', 'billion', 'trillion' and 'bn'. To be able to convert the column into numbers,
df_company.loc[:,'Total assets value'] = df_company.loc[:,'Total assets value'].str.replace('\strillion','e12', regex=True)  # replace trillion with e12
df_company.loc[:,'Total assets value'] = df_company.loc[:,'Total assets value'].str.replace('\sbillion','e9', regex=True)  # replace billion with e9
df_company.loc[:,'Total assets value'] = df_company.loc[:,'Total assets value'].str.replace('bn','e9', regex=True)  # replace bn with e9
df_company.loc[:,'Total assets value'] = df_company.loc[:,'Total assets value'].str.replace('\smillion','e6', regex=True)  # replace million with e6
df_company.loc[:,'Total assets value'] = df_company.loc[:,'Total assets value'].str.replace(',','', regex=True)  # deleting , due to english thousands separator notation
df_company.loc[:,'Total assets value'] = df_company.loc[:,'Total assets value'].astype(float)  # convert to float
df_company.loc[:,'Total assets value']

# strip() to remove spaces before or after the string
# Use the parameter na=False to replace the missing values.
mask_dollar = df_company.loc[:, 'Total assets'].str.strip().str.startswith('$', na=False)
mask_dollar

# Dollar to Euro
df_company.loc[mask_dollar, 'Total assets value'] = df_company.loc[mask_dollar, 'Total assets value'] * 0.86
df_company.loc[mask_dollar, 'Total assets value']


# The rest of the columns

for col in ['Operating income', 'Net income', 'Revenue', 'Total equity']:  # repeat for all columns with financial data
    print(col)
    df_company.loc[:, col+' year'] = df_company.loc[:, col].str.extract(r'(\d{4})', expand=False)  # extract the year
    df_company.loc[:, col+' value'] = df_company.loc[:, col].str.extract(r'([\d.,]+\s?[a-zA-Z]+)', expand =False)  # extract the value
    
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('\strillion','e12', regex=True)  # replace trillion with e12
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('\sbillion','e9', regex=True)  # replace billion with e9
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('bn','e9', regex=True)  # replace bn with e9
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('\smillion','e6', regex=True)  # replace million with e6
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('\sMio', 'e6', regex=True) # replace Mio with e6
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('M','e6', regex=True)  # replace M with e6
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('B','e9', regex=True)  # replace B with e9
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace('\seuro','', regex=True)  # replace , with .
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].str.replace(',','', regex=True)  # replace , with .
    df_company.loc[:, col+' value'] = df_company.loc[:, col+' value'].astype(float)  # convert to float
    
    mask_dollar = df_company.loc[:, col].str.strip().str.startswith('$', na=False)  # create mask to select the $-values
    df_company.loc[mask_dollar, col+' value'] = df_company.loc[mask_dollar, col+' value'] * 0.86  # calculate the €-values fromt the $-values
    

# Remember:

# Extract text in DataFrame columns with my_series.str.extract() using regexes
# Extract text from DataFrame columns with my_series.str.extract() using regexes
# Check whether texts in the DataFrame columns start with a certain string with my_series.str.startswith()
