# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:50:25 2017

@author: dcaramu
"""

import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('U:/Users/dcaramu/Desktop/Machine Learning Nano/1 spam detection/SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])
#Un ejemplo de como leer un csv: datos2=pd.read_csv("tabla_muestra.txt",sep=';',encoding="cp1252")


# Output printing out first 5 columns
df.head()

from pandasql import PandaSQL
pdsql = PandaSQL()

#Otra forma de hacer lo anterior
pdsql("SELECT * FROM df LIMIT 5;").head()

#cuantos registros tiene la tabla
pdsql("SELECT count(*) FROM df;")
#otra manera de hacer lo anterior pero tambien te muestra el número de columnas
print(df.shape)

#Convert the values in the 'label' colum to numerical values using map method as follows: {'ham':0, 'spam':1} This maps the 'ham' value to 0 and the 'spam' value to 1.

df['label'] = df.label.map({'ham':0, 'spam':1})
df.head() # returns (rows, columns)


#Step 2.1: Bag of words

#What we have here in our data set is a large collection of text data (5,572 rows of data). Most ML algorithms rely on 
#numerical data to be fed into them as input, and email/sms messages are usually text heavy.
#Here we'd like to introduce the Bag of Words(BoW) concept which is a term used to specify the problems that have a
#'bag of words' or a collection of text data that needs to be worked with. The basic idea of BoW is to take a piece of 
#text and count the frequency of the words in that text. It is important to note that the BoW concept treats each word 
#individually and the order in which the words occur does not matter.
#Using a process which we will go through now, we can covert a collection of documents to a matrix, with each document
#being a row and each word(token) being the column, and the corresponding (row,column) values being the frequency of 
#occurrance of each word or token in that document.

#To handle this, we will be using sklearns
#It tokenizes the string(separates the string into individual words) and gives an integer ID to each token.
#It counts the occurrance of each of those tokens.
#The CountVectorizer method automatically converts all tokenized words to their lower case form so that it does not 
#treat words like 'He' and 'he' differently. It does this using the lowercase parameter which is by default set to True.
#It also ignores all punctuation so that words followed by a punctuation mark (for example: 'hello!') are not treated
#differently than the same words not prefixed or suffixed by a punctuation mark (for example: 'hello'). It does this
#using the token_pattern parameter which has a default regular expression which selects tokens of 2 or more alphanumeric characters.
#The third parameter to take note of is the stop_words parameter. Stop words refer to the most commonly used words in a
#language. They include words like 'am', 'an', 'and', 'the' etc. By setting this parameter value to english, 
#CountVectorizer will automatically ignore all words(from our input text) that are found in the built in list of 
#english stop words in scikit-learn. This is extremely helpful as stop words can skew our calculations when we are 
#trying to find certain key words that are indicative of spam.

#Step 2.2: Implementing Bag of Words from scratch
#primero: hagamos un ejemplo con un documento cualquiera
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']
lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)
#segundo: removemos las puntuaciones
sans_punctuation_documents = []
import string
for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
print(sans_punctuation_documents) 
#tercero: tokenización
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)
#cuarto: contamos los tokens
frequency_list = []
import pprint
from collections import Counter
for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)
#Congratulations! You have implemented the Bag of Words process from scratch! 
#As we can see in our previous output, we have a frequency distribution dictionary which gives a clear view of the 
#text that we are dealing with.

#Step 2.3: Implementing Bag of Words in scikit-learn
#Now that we have implemented the BoW concept from scratch, let's go ahead and use scikit-learn to do this process in a
#clean and succinct way. We will use the same document set as we used in the previous step

#un pequeño ejemplo:
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
print(count_vector)
#it your document dataset to the CountVectorizer object you have created using fit(), and get the list of words which 
#have been categorized as features using the get_feature_names() method.
#The get_feature_names() method returns our feature names for this dataset, which is the set of words that make up 
#our vocabulary for 'documents'.
count_vector.fit(documents)
count_vector.get_feature_names()
#Create a matrix with the rows being each of the 4 documents, and the columns being each word. The corresponding 
#(row, column) value is the frequency of occurrance of that word(in the column) in a particular document(in the row).
#You can do this using the transform() method and passing in the document data set as the argument. The transform()
#method returns a matrix of numpy integers, you can convert this to an array using toarray(). 
doc_array = count_vector.transform(documents).toarray()
doc_array
#Now we have a clean representation of the documents in terms of the frequency distribution of the words in them.
#To make it easier to understand our next step is to convert this array into a dataframe and name the columns
#appropriately.
frequency_matrix = pd.DataFrame(doc_array, 
                                columns = count_vector.get_feature_names())
frequency_matrix
#Congratulations! You have successfully implemented a Bag of Words problem for a document dataset that we created.
#One potential issue that can arise from using this method out of the box is the fact that if our dataset of text is 
#extremely large(say if we have a large collection of news articles or email data), there will be certain values that 
#are more common that others simply due to the structure of the language itself. So for example words like 'is', 'the',
#'an', pronouns, grammatical contructs etc could skew our matrix and affect our analyis.
#There are a couple of ways to mitigate this. One way is to use the stop_words parameter and set its value to english.
#This will automatically ignore all words(from our input text) that are found in a built in list of English stop words
#in scikit-learn.

#Step 3.1: Training and testing sets
#Now that we have understood how to deal with the Bag of Words problem we can get back to our dataset and proceed with
#our analysis. Our first step in this regard would be to split our dataset into a training and testing set so we can
#test our model later.
#X_train is our training data for the 'sms_message' column.
#y_train is our training data for the 'label' column
#X_test is our testing data for the 'sms_message' column.
#y_test is our testing data for the 'label' column.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

#Step 3.2: Applying Bag of Words processing to our dataset.
# Instantiate the CountVectorizer method
count_vector = CountVectorizer()
# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

#Step 5: Naive Bayes implementation using scikit-learn
# We will be using sklearns sklearn.naive_bayes method to make predictions on our dataset. Specifically, we will be 
#using the multinomial Naive Bayes implementation. This particular classifier is suitable for classification with 
#discrete features (such as in our case, word counts for text classification). It takes in integer word counts as its 
#input. On the other hand Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data 
#has a Gaussian(normal) distribution.
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
#Now that predictions have been made on our test set, we need to check the accuracy of our predictions.

#Step 6: Evaluating our model
#Accuracy measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct 
#predictions to the total number of predictions (the number of test data points).
#Precision tells us what proportion of messages we classified as spam, actually were spam.
#[True Positives/(True Positives + False Positives)]
#Recall(sensitivity) tells us what proportion of messages that actually were spam were classified by us as spam.
#[True Positives/(True Positives + False Negatives)]

## IMPORTANTE:
#For classification problems that are skewed in their classification distributions like in our case, for example if we
#had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric.
#For such cases, precision and recall come in very handy. 
#These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. 
#This score can range from 0 to 1, with 1 being the best possible F1 score.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

#One of the major advantages that Naive Bayes has over other classification algorithms is its ability to handle an 
#extremely large number of features. In our case, each word is treated as a feature and there are thousands of 
#different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected 
#by them. The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box 
#and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is 
#known. It rarely ever overfits the data. Another important advantage is that its model training and prediction times 
#are very fast for the amount of data it can handle. All in all, Naive Bayes' really is a gem of an algorithm!