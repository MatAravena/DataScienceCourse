from best_book_data import *


# Command	Purpose
# import my_module	Imports the module my_module
# from my_module import my_object	Imports a single object from a module
# from my_module import *	Imports all objects from a module

import best_book_data
best_book_data??

# Capitalize names   .title()
very_good_titles[0] = very_good_titles[0].title()
print(very_good_titles[0])


# Congratulations: You have just recapped the str data type and two methods associated with it:

# my_str.title() (converts a str to a title-case version)
# You also got to know another list method:

# my_list.count() (returns the number of entries which have a certain value)
# Now let's move onto two other data types:

message = 'BestBooks currently has {} books in "{}" condition.\nThe average sale price for a "{}" book is {} USD, with a maximum price of {} USD and a minimum price of {} USD.'

good_count = len(good_titles)
good_sp_avg = sum(good_sp) / len(good_sp)
good_sp_max = max(good_sp)
good_sp_min = min(good_sp)

message.format(good_count, 'good', 'good', good_sp_avg, good_sp_max, good_sp_min)




condition = []
for i in range(len(book_arr)):
    if  i <= 10:
        condition.append(2)
    elif i >= 11 and i <= 26:
        condition.append(1)
    else:
        condition.append(0)
print(condition)

# Union arrays
book_arr = np.column_stack([book_arr, condition])
book_arr 





