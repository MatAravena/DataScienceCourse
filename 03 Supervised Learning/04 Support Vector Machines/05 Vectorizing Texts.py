# What the bag of words and term frequency - inverse document frequency methods do with texts.
# How to use bag of words and term frequency - inverse document frequency with sklearn.
# What the difference is between these two methods.



# vectorization 
# turn texts into usable numerical objects for machine learning algorithms
# (mathematically speaking, the final objects are vectors, the basic building blocks of linear algebra)


# vectorization 
# In NLP is the process of converting text to numbers 

# vectorization techniques: 
# Bag of words 
# term frequency - inverse document frequency.

import pandas as pd
df = pd.read_csv('text_messages.csv', index_col=0)
df.head()

import spacy
import nltk
import string
import re 
from nltk.corpus import stopwords 

stopWords = stopwords.words('english')
punctuations =  string.punctuation
nlp = spacy.load('en_core_web_sm')


# `text_cleaner` function
def text_cleaner(sentence):
    """Clean the text using typical NLP-Steps.
 
    Steps include: Lemmatization, removing stop words, removing punctuations 
 
    Args:
        sentence (str): The uncleaned text.
 
    Returns:
        str: The cleaned text.
        
    """

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
    return final_doc

df.loc[:, 'msg_clean'] = df.loc[:, 'msg'].apply(text_cleaner)
df.head()

# As we dont have another test data set we have to split the data train that we have

# train_test_split(X, #features (DataFrame or Series)
#                  y, #target (DataFrame or Series)
#                  test_size =float, #size of test set (between 0.1 and 1.0)
#                  random_state=int) #random seed generator (enables reproducibility)


# Variables result from train_test_split
# X_train, X_test, y_train, y_test

from sklearn.model_selection import train_test_split

features = df.loc[:, 'msg_clean']
target  = df.loc[:, 'status']

features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, 
                                                                            test_size = 0.3, 
                                                                            random_state = 1)


# The bag of words method
# The *bag of words* (BoW) method represents words in a **corpus** based on the frequency of the word in each text. 
# A BoW algorithm takes all the unique words present in the corpus and displays each text based on how often a word appears in that text. 
# The method gets its name from the fact that the order and structure of the words in the text are ignored.

print("MESSAGE 1 ORIGINAL: ", df.loc[660, 'msg'])
print("MESSAGE 1 CLEAN: ", df.loc[660, 'msg_clean'])
print("MESSAGE 1 TOKENIZED: ", [token for token in nlp(df.loc[660, 'msg_clean'])])

{"sea": 1, "lay": 1, "rock": 2, "envelope": 2, "paper": 2, "3": 1, "word": 1}


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()


# So we end up with what's called a sparse matrix. This is a matrix that contains the value 0 a lot. 
# To store them efficiently in the memory, you can use the datatype .csr_matrix (compressed sparse row matrix), which originally comes from the scipy module.

features_train_bow = count_vectorizer.fit_transform(features_train)
features_train_bow # <--  sparse matrix
# <3900x6333 sparse matrix of type '<class 'numpy.int64'>'
# 	with 32648 stored elements in Compressed Sparse Row format>


bow_features = count_vectorizer.get_feature_names()
len(bow_features)

# Conver spare matrix into a normal array
bow_array = features_train_bow.toarray()
bow_vector = pd.DataFrame(bow_array, columns=bow_features)
bow_vector.head()

# Checking data
mask = bow_vector.loc[:,'call'] != 0
bow_vector.loc[mask,'call']





# The term frequency - inverse document frequency method
# takes the relevance of a word into account.

# To determine the relevance of a word, TF-IDF takes two things into account, as its name suggests:

# How often a word appears in an individual message, based on the number of words in the same message (term frequency, the TF in TF-IDF).
# TF(𝐰𝐨𝐫𝐝,𝐦𝐞𝐬𝐬𝐚𝐠𝐞)=numberof instances of a 𝐰𝐨𝐫𝐝 ina 𝐦𝐞𝐬𝐬𝐚𝐠𝐞 / numberofallwordsinthesame𝐦𝐞𝐬𝐬𝐚𝐠𝐞
 
# How many messages contain a certain word, in relation to the size of the corpus (inverse document frequency, the IDF in TF-IDF).
# IDF(𝐰𝐨𝐫𝐝,𝐜𝐨𝐫𝐩𝐮𝐬)=ln(numberof𝐦𝐞𝐬𝐬𝐚𝐠𝐞𝐬inthe𝐜𝐨𝐫𝐩𝐮𝐬 / numberof𝐦𝐞𝐬𝐬𝐚𝐠𝐞𝐬,thatcontainthe𝐰𝐨𝐫𝐝)
 
# This is a natural logarithm. The TF-IDF value of a word in a message in the corpus is defined as the product of TF and IDF.

# TF-IDF(𝐰𝐨𝐫𝐝,𝐦𝐞𝐬𝐬𝐚𝐠𝐞,𝐜𝐨𝐫𝐩𝐮𝐬)=TF(𝐰𝐨𝐫𝐝,𝐦𝐞𝐬𝐬𝐚𝐠𝐞)⋅IDF(𝐰𝐨𝐫𝐝,𝐜𝐨𝐫𝐩𝐮𝐬)


# The more messages the word appears in, the less valuable this word is for differentiating between text types. 
# The TF-IDF of this kind of word would be small. An important word would be one that occurs very rarely in the entire corpus.


# Tokenizing Message 1
message_1_tokenized = [token for token in nlp(df.loc[660, 'msg_clean'])]
print("MESSAGE 1 TOKENIZED: ", message_1_tokenized)

# Tokenizing Message 2
message_2_tokenized  = [token for token in nlp(df.loc[1178, 'msg_clean'])]
print("MESSAGE 2 TOKENIZED: ", message_2_tokenized)

# Tokenizing Message 3
message_3_tokenized  = [token for token in nlp(df.loc[621, 'msg_clean'])]
print("MESSAGE 3 TOKENIZED: ", message_3_tokenized)




# Calculating TF
# Length of tokenized Message 1
message_1_tokenized_len = len(message_1_tokenized)
print("NUMBER OF WORDS IN MESSAGE 1: ", message_1_tokenized_len)

# Calculating TF of `word` which appears once in the first message
message_1_tf_word = 1 / message_1_tokenized_len

# Calculating TF of `rock` which appears twice in the first message 
message_1_tf_rock = 2 / message_1_tokenized_len

print("TF OF 'word' IN MESSAGE 1: ", message_1_tf_word)
print("TF OF 'rock' IN MESSAGE 1: ", message_1_tf_rock)

# Length of tokenized Message 2
message_2_tokenized_len = len(message_2_tokenized)
print("NUMBER OF WORDS IN MESSAGE 2: ", message_2_tokenized_len)

# Calculating TF of `run` which appears once in the second message
message_2_tf_run = 1 / message_2_tokenized_len

# Calculating TF of `rock` which appears once in the second message
message_2_tf_rock = 1 / message_2_tokenized_len

print("TF OF 'run' IN MESSAGE 2: ", message_2_tf_run)
print("TF OF 'rock' IN MESSAGE 2: ", message_2_tf_rock)

# Length of tokenized Message 3
message_3_tokenized_len = len(message_3_tokenized)
print("NUMBER OF WORDS IN MESSAGE 3: ", message_3_tokenized_len)

# Calculating TF of `word` which appears twice in the third message
message_3_tf_word = 2 / message_3_tokenized_len

print("TF OF 'word' IN MESSAGE 3: ", message_3_tf_word)


# Calculating IDF
import math
# There are 3 messages in our corpus

# 'rock' appears in messages 1 and 2
idf_rock = math.log(3 / 2)

# 'word' appears in messages 1 and 3
idf_word = math.log(3 / 2)

# 'run' appears in message 2
idf_run = math.log(3 / 1)

print("IDF of 'rock': ", idf_rock)
print("IDF of 'word': ", idf_word)
print("IDF of 'run': ", idf_run)



# Calculating TF-IDF
print("TF-IDF of 'rock' in Message 1: ", message_1_tf_rock * idf_rock)
print("TF-IDF of 'rock' in Message 2: ", message_2_tf_rock * idf_rock)

print("TF-IDF of 'word' in Message 1: ", message_1_tf_word * idf_word)
print("TF-IDF of 'word' in Message 3: ", message_3_tf_word * idf_word)

print("TF-IDF of 'run' in Message 2: ", message_2_tf_run * idf_run)


# now with Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

features_train_tfidf = tfidf_vectorizer.fit_transform(features_train)

tfidf_features = tfidf_vectorizer.get_feature_names() 
len(tfidf_features)

tfidf_vector = pd.DataFrame(features_train_tfidf.toarray(), columns = tfidf_features)
tfidf_vector.head()


idf_values = tfidf_vectorizer.idf_
print(min(idf_values))
print(max(idf_values))

print("IDF VALUE OF CALL: ", idf_values[tfidf_features.index("call")])


# Comparing BoW and TF-IDF
print(bow_vector.loc[:12, 'call'])
print(tfidf_vector .loc[:12, 'call'])

# Messages
print("Message 3: ", features_train.iloc[3])
print("Message 7: ", features_train.iloc[7])

# TF-IDF considers the fact that message 7 only has three tokens. This means that in relation to message 3, 
# the word "call" is more important for understanding the meaning of message 7 than in message 3. 
# This example shows how TF-IDF captures subtle nuances not found in BoW.


# Remember:

# The BoW method (CountVectorizer) vectorizes words based on their frequencies in the text.
# TF-IDF vectorization (TfidfVectorizer) takes into account the relevance of a word.

