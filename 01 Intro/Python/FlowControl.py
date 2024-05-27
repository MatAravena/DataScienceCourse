

full_names = []
for i in range(len(given_names)):
    full_names.append(given_names[i] + ' ' + family_names[i])
print(full_names)




full_names = [given_names[i] + ' ' + family_names[i] for i in range(len(given_names))]
print(full_names)

#var_list = [list_element for looping_variable in var_sequence if conditional_statement]
list_sp_like_new = []
list_sp_like_new = [ float(book[5]) for book in book_arr if book[6] == '2']

sum(list_sp_like_new)/len(list_sp_like_new)



# Remember:

# if statement
# if condition :
#   do_something

# for loop
# for looping_variable in var_sequence:
#   do_something

# List comprehension
# var_list = [list_element for looping_variable in var_sequence if conditional_statement]
# Indentation is an important part of Python syntax
