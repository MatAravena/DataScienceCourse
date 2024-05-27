# Prepare text data for text classification tasks.
# Use different Python modules for preparing text data.

# Natural Language Processing


import pandas as pd
df = pd.read_csv('text_messages.csv', index_col=0)
df.head()

# A DataFrame with two columns:
# which contains the text messages, and 'status', which indicates whether it is a normal text message or whether it is spam.
# In NLP this kind of collection of text data is called a corpus.

print(df.loc[122, 'msg'])
print(df.loc[105, 'msg'])

# How to identify which is a spam message or a normal text
import string
import re

# Python modles for NLP, some of all of them
import spacy
import nltk

# Functions from nltk accept str values and output str values. In contrast, spacy uses an object-oriented approach and usually 
# returns document objects with their own attributes and methods. Many users find spacy to be more time and memory efficient 
# than nltk and therefore more suitable for production.

from spacy.lang.en.examples import sentences
nlp = spacy.load('en_core_web_sm')
print(type(nlp))
# <class 'spacy.lang.en.English'>


# Tokenization
# A .Doc object is a sequence of token objects, which are the individual linguistic units we need for our analysis


# print original message
print(df.loc[105, 'msg'])
print(type(df.loc[105, 'msg']))

# create a doc variable
doc = nlp(df.loc[105, "msg"])
print(doc)

# check data type of doc
type(doc)


doc_tokens = [token.text for token in doc]
print(doc_tokens)
# ['Todays', 'Voda', 'numbers', 'ending', '7548', 'are', 'selected', 'to', 'receive', 'a', '$', '350', 'award', '.', 'If', 'you', 'have', 'a', 'match', 'please', 'call', '08712300220', 'quoting', 'claim', 'code', '4041', 'standard', 'rates', 'app']

# POS tagging
token_pos = [[token.text, token.pos_] for token in doc]
print(token_pos)
# [['Thanks', 'NOUN'], ['a', 'DET'], ['lot', 'NOUN'], ['for', 'ADP'], ['your', 'PRON'], ['wishes', 'NOUN'], ['on', 'ADP'], ['my', 'PRON'], ['birthday', 'NOUN'], ['.', 'PUNCT'], ['Thanks', 'NOUN'], ['you', 'PRON'], ['for', 'ADP'], ['making', 'VERB'], ['my', 'PRON'], ['birthday', 'NOUN'], ['truly', 'ADV'], ['memorable', 'ADJ'], ['.', 'PUNCT']]



# Cleaning and preparing text data

# Some cleaning and preparation techniques are:
# Lemmatization
# Removing stop words
# Removing punctuation marks

# Lemmatization means that words are reduced to their root form, also known as a lemma.

# print original message
print("ORIGINAL TEXT:\n", doc)

# get the lemmas of doc
lemma_token = [token.lemma_ for token in doc]
print("AFTER LEMMATIZATION:\n", lemma_token)
# ORIGINAL TEXT:
#  Thanks a lot for your wishes on my birthday. Thanks you for making my birthday truly memorable.
# AFTER LEMMATIZATION:
#  ['thank', 'a', 'lot', 'for', 'your', 'wish', 'on', 'my', 'birthday', '.', 'thank', 'you', 'for', 'make', 'my', 'birthday', 'truly', 'memorable', '.']

# Removing pronoumns from the phrase
lemma_token = [token.lemma_ for token in doc if token.pos_ != "PRON"]
print('\n')
print(lemma_token)



# Removing stop words
# is a way to remove common words from a text. Stop words are generally articles ("the" and "a"), pronouns such as "I" and "you" (which we already removed in the previous step) or common verbs ("be", "can"). These words appear frequently in most English language texts. Removing these words reduces the amount of data that needs to be analyzed, while allowing machine learning algorithms to put more emphasis on tokens that give a text its true meaning.


# Stop words provided in the nltk module
# import stopwords from nltk
from nltk.corpus import stopwords

# convert stopwords to set
stopWords = stopwords.words('english')

# print stopwords
print(stopWords)


# We now want to remove all the stop words from lemma_token
# print message
print("LEMMATIZED TEXT:\n", lemma_token)

# remove stopwords and print again
no_stopWords_lemma_token = [token.lower() for token in lemma_token if token not in stopWords]
print("NO STOP WORDS:\n", no_stopWords_lemma_token)




# removing punctuation
# involves removing punctuation marks and symbols that do not contribute to the meaning of the text.

# save punctuations and print them
punctuations = string.punctuation
print(punctuations)
# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# print message
print("TEXT WITH NO STOP WORDS:\n", no_stopWords_lemma_token)

# remove punctuations
clean_doc = [token for token in no_stopWords_lemma_token if token not in punctuations]
print("NO PUNCTUATIONS:\n", clean_doc)




# Creating a .Doc variable `doc` using `nlp()` for message with line number 3202
doc_new = nlp(df.loc[3202, "msg"])

# Lemmatization: creating a list of the attributes from my_token.lemma_
# Using an if-request in order to exclude 'PRON'
new_lemma_token = [token.lemma_ for token in doc_new if token.pos_ != "PRON"]

# Removing stopwords: creating a list without words that are contained in stopWords and converting tokens to lowercase
new_lemma_token = [token.lower() for token in new_lemma_token if token not in stopWords]

# Removing punctuations: Creating a list without strings that are contained in punctuations
new_lemma_token = [token for token in new_lemma_token if token not in punctuations]

# print clean_doc
print(new_lemma_token)



# A user-defined function to clean and prepare text
def text_cleaner(sentence):
    # Create the Doc object named `text` from `sentence` using `nlp()`
    doc = nlp(sentence)
    # Lemmatization
    lemma_token = [token.lemma_ for token in doc if token.pos_ != 'PRON']
    # Remove stop words and converting tokens to lowercase
    no_stopWords_lemma_token = [token.lower() for token in lemma_token if token not in stopWords]
    # Remove punctuations
    clean_doc = [token for token in no_stopWords_lemma_token if token not in punctuations]
    # Output    
    return clean_doc




def text_cleaner(sentence):
    # Create the Doc object named `text` from `sentence` using `nlp()`
    doc = nlp(sentence)
    # Lemmatization
    lemma_token = [token.lemma_ for token in doc if token.pos_ != 'PRON']
    # Remove stop words and converting tokens to lowercase
    no_stopWords_lemma_token = [token.lower() for token in lemma_token if token not in stopWords]
    # Remove punctuations
    clean_doc = [token for token in no_stopWords_lemma_token if token not in punctuations]
    
    # Use the `.join` method on `text` to convert string
    joined_clean_doc = " ".join(clean_doc)
    # Use `re.sub()` to substitute multiple spaces or dots`[\.\s]+` to single space `' '
    final_doc = re.sub('[\.\s]+', ' ', joined_clean_doc)
    
    # Output    
    return final_doc



# Remember:

# Text data first has to be cleaned and prepared before it can be analyzed.
# Some common text cleaning tasks for NLP are lemmatization, as well as removing stop words and punctuation.
# Some helpful modules for NLP are spacy, nltk, string and re.