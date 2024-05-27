# Understand how different vectorization methods produce different results.
# Combine SVC with CountVectorizer, TfidfVectorizer and GridSearch



## Cleaning and vectorizing text data

import pandas as pd
df = pd.read_csv('text_messages.csv', index_col=0)
df.head()

import spacy
import nltk
import string
import re

# Import stopwords from nltk
from nltk.corpus import stopwords

# Save stopwords as a set
stopWords = set(stopwords.words('english'))

# List of punctuation marks
punctuations = string.punctuation

# Loading the English module
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

# # Preparing test data

from sklearn.model_selection import train_test_split

features = df.loc[:, 'msg_clean']  # features are based on cleaned dataset
target = df.loc[:, 'status']

features_train, features_test, target_train, target_test = train_test_split(features, 
                                                                            target, 
                                                                            test_size=0.3, # use 30% of data as test set
                                                                            random_state=1)

# # Vectorizing
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()

features_train_bow = count_vectorizer.fit_transform(features_train)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

features_train_tfidf = tfidf_vectorizer.fit_transform(features_train)


# # # Instantiating the LinearSVC
# This is a version of SVC specially optimized for linear kernels and you should always use it instead of SVC with kernel='linear'.
from sklearn.svm import LinearSVC
model = LinearSVC(dual=False , random_state=1)

from sklearn.model_selection import GridSearchCV 

search_space = { "C": [0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced']  }

model_grid = GridSearchCV(estimator=model,
                        param_grid=search_space, 
                        n_jobs=-1,
                        cv= 5,
                        scoring='f1')



# # Using SVC on the BoW features
model_grid.fit(features_train_bow, target_train)

print('best f1 score',model_grid.best_score_ )
print('best C',model_grid.best_estimator_.C )
print('best class weight',model_grid.best_estimator_.class_weight)

features_test_bow = count_vectorizer.transform(features_test)

target_test_pred_bow = model_grid.predict(features_test_bow)

from sklearn.metrics import confusion_matrix
confusion_matrix_bow = confusion_matrix(target_test, target_test_pred_bow)

import seaborn as sns

sns.heatmap(confusion_matrix_bow, annot=True, fmt='d')  # annot=True shows the numbers, fmt='d' supresses scientific notation

from sklearn.metrics import classification_report
print(classification_report(target_test, target_test_pred_bow, target_names=["Not spam", "Spam"]))  # We need print because classification_report returns a string with linebreaks

#               precision    recall  f1-score   support

#     Not spam       0.99      1.00      0.99      1454
#         Spam       0.98      0.90      0.94       218

#     accuracy                           0.98      1672
#    macro avg       0.98      0.95      0.96      1672
# weighted avg       0.98      0.98      0.98      1672

# support is the number of data points that belong to the corresponding class. 
# Then we're shown the accuracy and two different average values for all the other metrics: The macro average and weighted average.





# # Predictions with the TF-IDF features
model_grid.fit(features_train_tfidf, target_train)

print('best f1 score',model_grid.best_score_)
print('best C',model_grid.best_estimator_.C)
print('best class weight',model_grid.best_estimator_.class_weight)

features_test_tfidf = tfidf_vectorizer.transform(features_test)

target_test_pred_tfidf = model_grid.predict(features_test_tfidf)

confusion_matrix_bow = confusion_matrix(target_test, target_test_pred_tfidf)

sns.heatmap(confusion_matrix_tfidf, annot=True, fmt='d')  # annot=True shows the numbers, fmt='d' supresses scientific notation

print(classification_report(target_test, target_test_pred_tfidf, target_names=["Not spam", "Spam"]))

#               precision    recall  f1-score   support

#     Not spam       0.99      1.00      0.99      1454
#         Spam       0.97      0.93      0.95       218

#     accuracy                           0.99      1672
#    macro avg       0.98      0.96      0.97      1672
# weighted avg       0.99      0.99      0.99      1672



# # Improvements with different kernels?
from sklearn.svm import SVC

model_kernel = SVC(random_state=1, max_iter=10000)

search_space_kernel = {'C': [0.01, 0.1, 1, 10, 100],
                       'kernel': ['poly', 'rbf'],
                       'class_weight': [None, 'balanced']}

model_grid_kernel = GridSearchCV(estimator=model_kernel,
                                 param_grid=search_space_kernel,
                                 cv=5,
                                 n_jobs=-1,
                                 scoring="f1")

model_grid_kernel.fit(features_train_bow,target_train)

print('Best score:', model_grid_kernel.best_score_)
print('Best C:',model_grid_kernel.best_estimator_.C)
print('Best kernel:', model_grid_kernel.best_estimator_.kernel)
print('Best class_weight:',model_grid_kernel.best_estimator_.class_weight)


model_grid_kerneltfidf = GridSearchCV(estimator=model_kernel,
                                 param_grid=search_space_kernel,
                                 cv=5,
                                 n_jobs=-1,
                                 scoring="f1")

model_grid_kerneltfidf.fit(features_train_tfidf,target_train)

print('Best score:', model_grid_kerneltfidf.best_score_)
print('Best C:',model_grid_kerneltfidf.best_estimator_.C)
print('Best kernel:', model_grid_kernel.best_estimator_.kernel)
print('Best class_weight:',model_grid_kerneltfidf.best_estimator_.class_weight)



# Remember:

# The way texts are vectorized and the choice of kernel has an influence on the prediction quality
# Generate classification report with classification_report(target, target_pred).