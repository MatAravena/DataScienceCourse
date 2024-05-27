# Read PDF files
# Extract data from text using regular expressions


# Reading PDF documents

import PyPDF2
# Argument	        Meaning	        Use
# 'r' (default)	    read	        When you want to read the file without changing it
# 'w'	            write	        When you want to rewrite/overwrite the file
# 'a'	            add	            When you want to add new characters at the end of the file
# 'rb', 'wb', 'ab'	binary format	If the file doesn't use simple text encoding

file_reader = open('invoice.pdf', 'rb')

PyPDF2.PdfFileReader(file_reader)
pdf_reader = PyPDF2.PdfFileReader(file_reader)

pdf_reader.getNumPages()

pdf_page = pdf_reader.getPage(0)

pdf_str = pdf_page.extractText()
pdf_str[:100]

file_reader.close()



# Regular expressions

pdf_str = pdf_str.replace('\n','')
pdf_str

import re    #--> Regular expression operations

expression = 'Customer-ID'  # define regular expression
re.findall(expression, pdf_str)  # look for regular expression in pdf_str


# But the strength of regular expressions is that they can generalize characters to actually search for common patterns
expression = r'Customer-ID\.:\s\d+'  # define regular expression
re.findall(expression ,pdf_str)  # look for regular expression in pdf_str


# In regular expressions, 
# '\' is the escape character: it prevents the following character from being interpreted 
# '+' repetitions in our search pattern is for. It specifies that the preceding character (a digit '\d') should occur at least once


# The same 
re.findall(r'\d\d\.\d\d\.\d\d\d\d',pdf_str)
re.findall(r'\d+\.\d+\.\d+',pdf_str)

expression = r'Invoice No\.\s\d+\-\d+\-\d+'  # define regular expression
re.findall(expression, pdf_str)

# Find number
invoice_price = re.findall(r'Total .* EUR', pdf_str)[0]
invoice_price = invoice_price.split(' ')[1].replace(',', '.')  # split the price and replace , with .
invoice_price = float(invoice_price)
print(invoice_price)


# Email
expression = r'\w+@\w+\.\w{2}'
invoice_mail = re.findall(expression, pdf_str)[0]
invoice_mail

# expression = r'\b\d{5}\s[^\d\s]+'
# re.findall(expression, pdf_str)

# Make a list
expression = r'(\b\d{5})\s([^\d\s]+)'
re.findall(expression, pdf_str)

expression = r'(\b\d{5})\s([^\d\s]+)'  # define regex with capture groups
post_and_city = re.findall(expression, pdf_str)[1]  # select the element containing München
invoice_post_code = post_and_city[0]
invoice_city = post_and_city[1]
invoice_city


# Remember:

# Get text from a PDF in the following steps:
# Open the document: my_file_reader = open('my_file_path', 'my_mode')
# Open it with PyPDF2: my_pdf_reader = PyPDF2.PdfFileReader(my_file_reader)
# Select page: my_pdf_page = my_pdf_reader.getPage(my_page_number)
# Extract the text: my_pdf_page.extractText()
# Regular expressions are search patterns for texts
# Extract search patterns with re.findall(my_expression, my_str)