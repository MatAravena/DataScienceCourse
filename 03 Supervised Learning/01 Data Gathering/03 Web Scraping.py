# You will know what to look for for when web scraping
# You will be able to easily iterate through the elements of several lists
# You will be able to search specifically for classes in HTML


# Gathering information from multiple websites
# connect to the website
import requests
from bs4 import BeautifulSoup  # import parser

website_url = 'https://en.wikipedia.org/wiki/DAX'
response = requests.get(website_url)  # get content of website
response.raise_for_status()  # give error if request failed

# get the table
soup = BeautifulSoup(response.text) # parse website
table = soup.find(id='constituents')  # select the table using the id

links_wiki = [link.attrs['href'] for link in table.find_all('a') if not 'class' in link.attrs]
links_wiki[:5]

links_wiki.remove('/wiki/Prime_Standard')
links_wiki.remove('#endnote_1')
links_wiki[:5]

link = links_wiki[0]
print(link)
responseLink = requests.get(base_wiki+link)

import pandas as pd
df_table = pd.read_html(response.text, attrs={'class': 'infobox vcard'})[0]

# Delete empty rows
df_table = df_table.loc[~df_table[0].isna(),:]
df_table = df_table.iloc[:, :2]

company_name = link.split('/')[-1]
company_name

f_table.columns=['key',company_name]
df_table = df_table.set_index('key')
df_table




def load_link(link, base_url='https://en.wikipedia.org'):
    """Extracts information in table with class 'infobox vcard' from a wikipedia link. 
    Returns a single column DataFrame with basic company information.
  
    Args:
        link (str): Subpage of wikipedia.
        base_url (str): Wikipedia main page defaults to 'https://en.wikipedia.org'.
 
    Returns:
        DataFrame: Shape (x,1) with column name [company_name] and keys as index.
    """
    #connect to page
    response = requests.get(base_url+link)
    
    #raise error if no connection
    response.raise_for_status()
    
    #extact table
    dfTable = pd.read_html(response.text,attrs={'class':'infobox vcard'})[0]
    
    #clean table
    dfTable = dfTable.loc[~dfTable[0].isna()]
    dfTable = dfTable.iloc[:,:2]
    company_name = link.split('/')[-1]
    dfTable.columns=['key',company_name]
    dfTable = dfTable.set_index('key')

    #add delay
    time.sleep(response.elapsed.total_seconds()*5)
    
    #remove duplicated values
    dfTable = dfTable.loc[~dfTable.index.duplicated(keep='last')]

    return dfTable



# Check if is ok the method
(load_link(links_wiki[0]) != df_table).sum()

company_dfs = []
for link in links_wiki:
    print(link)
    df = load_link(link)
    company_dfs.append(df)

# pd.concat(objs=list # list with DataFrames or Series
#           axis=int # 0: concatenates all rows together one below the other 1: concatenates all columns side by side
#          )

df_company = pd.concat(company_dfs, axis=1)
df_company.head()


# To change from a wide format to a long format, you simply need to transpose your DataFrame with my_df.T
df_company = df_company.T
df_company.head()




df_company.isna().sum()

cols = ['Revenue', 'Operating income', 'Net income', 'Total assets', 'Total equity']
df_company.loc[:,cols]


df_company.to_pickle('company_data.p')


# Remember:

# Combine several DataFrames (or Series) with pd.concat() to make a bigger DataFrame
# Don't send too many queries in a short period of time. You can use sleep() from the time module here.
# Pay attention to similarities and differences in the web pages when reading them out
# Web pages change their content and structure regularly and sometimes unexpectedly. Because of this, parsers need to be adjusted regularly!