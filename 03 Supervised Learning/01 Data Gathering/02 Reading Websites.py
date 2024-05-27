# You will know how HTML documents are structured
# You will be able to access HTML code with the beautifulsoup module

import requests
website_url  = 'https://en.wikipedia.org/wiki/DAX'

response = requests.get(website_url)
response.status_code == requests.codes.ok

response.text.count('<table')

# Reading HTML documents
# HTML parsers --> beautifulsoup

from bs4 import BeautifulSoup

# BeautifulSoup(markup=str #String containing website HTML code)

soup = BeautifulSoup(response.text)

print(soup.prettify()[:1000])
tables = soup.find_all('table')

for table in tables:
    print(table.attrs)

table = soup.find(id='constituents')

table_list = []
for row in table.find_all('tr'):
    table_list.append(row.text.split('\n'))
print(table_list)

for row in table_list:
    print(len(row))

import pandas as pd
df_dax = pd.DataFrame(table_list[1:], columns=table_list[0])
print(df_dax.shape)
df_dax.head()

df_dax = df_dax.drop('', axis=1)
df_dax.head()
 
df_dax.info()

df_dax.loc[:, 'Employees'] = df_dax.loc[:, 'Employees'].str.replace(r'\(\d\d\d\d\)', '', regex=True)

df_dax.loc[:, 'Index weighting (%)1'] = pd.to_numeric(df_dax.loc[:, 'Index weighting (%)1'], errors='coerce')
df_dax.loc[:, 'Employees'] = pd.to_numeric(df_dax.loc[:, 'Employees'].str.replace(',', ''))
df_dax.loc[:, 'Founded'] = pd.to_numeric(df_dax.loc[:, 'Founded'])

df_dax.dtypes
df_dax.to_pickle('dax_data.p')

# pd.read_html(io=str,     #url or HTML text
#              attrs=dict) #(optional) Attributes to select specific table like {attr: value}

dfs = pd.read_html(website_url, attrs={'id': 'constituents'})
dfs[0].head()


# Remember:
# Transform HTML code with beautifulsoup
# Iterate through elements with a certain tag with my_soup.find_all('tagname')
# Find specific elements with id attribute with my_soup.find(id='my_id')
# Output all text content within an element and its sub-elements with my_element.text
# Read web pages with the help of pd.read_html()
