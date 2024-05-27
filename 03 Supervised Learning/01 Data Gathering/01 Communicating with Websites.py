# The requests module
# Website status codes


# web scraping
import requests
website_url = 'https://en.wikipedia.org/wiki/DAX'
response = requests.get(website_url)
response

response.status_code == requests.codes.ok
# True

response_404 = requests.get('https://httpbin.org/status/404')
response_404.raise_for_status()

response.raise_for_status()

print(response.headers['content-type'])
print(response.headers['date'])


# When structuring data, you should pay attention to 3 principles to ensure that the data can be used as easily as possible for automated evaluations and models. 
# These principles are often called *tidy data principles*. They are as follows:

# * Each observation has its own row
# * Each variable has its own column
# * Each value has its own cell

# Remember:

# Access a web page with requests.get('my_website_url')
# Check the status of the server response with my_response.status_code == requests.codes.ok or generate an error with my_response.raise_for_status() if the response is unusable.
# Output HTML content of the web page with my_response.text