print(book_arr.shape)

titles= [book_arr[i][0] for i in range(len(book_arr))]
authors = [book_arr[i][1] for i in range(len(book_arr))]
pubyear = [book_arr[i][2] for i in range(len(book_arr))]
book_format = [book_arr[i][3] for i in range(len(book_arr))]
list_price = [book_arr[i][4] for i in range(len(book_arr))]
sale_price = [book_arr[i][5] for i in range(len(book_arr))]
condition = [book_arr[i][6] for i in range(len(book_arr))]

book_dict = {
    'titles' : titles,
    'authors' : authors,
    'pubyear' : pubyear,
    'book_format' : book_format,
    'list_price' : list_price,
    'sale_price' : sale_price,
    'condition' : condition
}

book_dict= {}

for key in range(len(book_dict)):
    [book_arr[i][key] for i in range(len(book_arr))]

book_dict = {}
for i in range(len(book_arr)) :
    book_dict[i] = {
        'titles' : book_arr[i][0],
        'authors' : book_arr[i][1],
        'pubyear' : book_arr[i][2],
        'book_format' : book_arr[i][3],
        'list_price' : book_arr[i][4],
        'sale_price' : book_arr[i][5],
        'condition' : book_arr[i][6]
    }

for key in book_dict.keys():
    print(book_dict[key])

book_store_items = []
for i in range(len(book_dict)):
    #print(book_dict[i])
    #print(range(len(book_dict[i])))
    #temp_list = [key for key in range(len(book_dict[i]))]
    #print('Key', key, [book_dict[key] for key in range(len(book_dict[i]))] )
    book_store_items.append(book_dict[i])

print(book_store_items)
print(len(book_store_items))

# Remember:

# dict are organized with key-value pairs
# You access values in a dict with square brackets.
# my_dict[my_key]
# The my_dict.update({my_key:my_value}) method allows you to update existing entries in a dict or add new entries if they aren't already in the dict.



df.groupby('Condition')['Sale Price'].unique()


# Remember:

# DataFrames from the pandas module are the best data type for structured data.
# Group data with my_df.groupby() and aggregate it with agg() and the function.
# Select data that meets a condition using boolean masks.