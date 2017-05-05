
# coding: utf-8

# In[23]:

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif


# In[24]:

df = pd.read_csv('/Users/kazhitu/Documents/YelpDataset_Reviews_Stars_20K.csv', header = None)
ldf = pd.read_csv('/Users/kazhitu/Documents/YelpDataset_Reviews_Stars_20K.csv', header = None)


# In[25]:

print(df)


# In[26]:

def filterLen(docs, minlen=3):
    import re, string
    docs_raw = [re.sub('[\W_]+','',t) for t in docs.split() if len(t) >= minlen]
    s = " "
    return s.join(docs_raw)


# In[27]:

import re
df[1] = df[1].apply(filterLen)
ldf[1] = ldf[1].apply(filterLen)


# In[28]:

xtrain, xtest, ytrain, ytest = df[1][:15000], df[1][15000:], df[0][:15000], df[0][15000:]


# In[29]:

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
def lemma(docs):
    wnl = WordNetLemmatizer()
    word = " "
    return word.join([wnl.lemmatize(i) for i in docs.split()])


# In[30]:

import nltk
from nltk.corpus import stopwords

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 


# In[31]:

clean_Train = review_to_words( xtrain[0] )
print(xtrain[0])
print(clean_Train)


# In[32]:

ldf[1] = df[1].apply(lemma)


# In[33]:

xtrainlemma, xtestlemma, ytrainlemma, ytestlemma = ldf[1][:15000], ldf[1][15000:], df[0][:15000], df[0][15000:]


# In[34]:

clean_Train2 = review_to_words(xtrainlemma[0])
print(clean_Train2)


# In[35]:

num_train_reviews = xtrain.size
num_test_reviews = xtest.size
print(num_train_reviews, num_test_reviews)


# In[36]:

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_train_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( xtrain[i] ) )


# In[37]:

from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()


# In[38]:

print train_data_features.shape


# In[39]:

vocab = vectorizer.get_feature_names()
print vocab


# In[40]:

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag


# In[41]:

print(xtest.shape)
print(xtest[15000])


# In[42]:

# Initialize an empty list to hold the clean reviews
clean_test_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_test_reviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_test_reviews.append( review_to_words( xtest[i+15000] ) )


# In[43]:

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


# In[44]:

print test_data_features.shape


# In[45]:

vocab = vectorizer.get_feature_names()
print vocab


# In[46]:

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(test_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag


# In[47]:

print test_data_features
print train_data_features


# In[48]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(train_data_features, ytrain).predict(test_data_features)


# In[49]:

print (y_pred[:300])
print (ytest[:300])


# In[50]:

from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, ytrain )


# In[51]:

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)
print (result[:300])
print (ytest[:300])


# In[52]:

#The correctness of random forest
correct = 0
for i in range(0,4999):
    if (ytest[i+15000] == result[i]):
        correct += 1
acc = correct/5000.00
print acc


# In[53]:

#The correctness of naive bayes
correct = 0
for i in range(0,4999):
    if (ytest[i+15000] == y_pred[i]):
        correct += 1
acc = correct/5000.00
print "Accuracy of Naive bayes w/ countvectorizer: %f" %acc


# In[54]:

#naive bayes
err = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range (0,4999):
    if (ytest[i+15000] != y_pred[i]):
        err[y_pred[i] - ytest[i+15000] + 4] += 1

print "Error distribution of Naive bayes w/ countvectorizer:"
print err


# In[55]:

import numpy as np
import matplotlib.pyplot as plt

bar_width = 1. # set this to whatever you want
data = np.array(err)
positions = np.arange(9)
plt.bar(positions, data, bar_width)
plt.xticks(positions + bar_width / 2, ('-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'))
plt.show()


# In[56]:

#random forest
err = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range (0,4999):
    if (ytest[i+15000] != result[i]):
        err[result[i] - ytest[i+15000] + 4] += 1    
print err


# In[57]:

bar_width = 1. # set this to whatever you want
data = np.array(err)
positions = np.arange(9)
plt.bar(positions, data, bar_width)
plt.xticks(positions + bar_width / 2, ('-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'))
plt.show()


# In[58]:

#random forest 
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 300) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, ytrain )
result = forest.predict(test_data_features)
#The correctness of random forest
correct = 0
for i in range(0,4999):
    if (ytest[i+15000] == result[i]):
        correct += 1
acc = correct/5000.00
print acc


# In[59]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#vectorizer = TfidfVectorizer(min_df=1)
#X = vectorizer.fit_transform(clean_train_reviews)
#idf_train = vectorizer.toarray()
#print(idf_train)
transformer = TfidfTransformer(smooth_idf=False)
trainidf = transformer.fit_transform(train_data_features)
trainidf = trainidf.toarray()
print(trainidf)


# In[60]:

#vectorizer = TfidfVectorizer(min_df=1)
#X = vectorizer.fit_transform(clean_test_reviews)
#idf_test = vectorizer.idf_
#print(idf_test)
transformer = TfidfTransformer(smooth_idf=False)
testidf = transformer.fit_transform(test_data_features)
testidf = testidf.toarray()
print(testidf)


# In[61]:

print(testidf.shape)
print(trainidf.shape)


# In[62]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred2 = gnb.fit(trainidf, ytrain).predict(testidf)


# In[63]:

#The correctness of naive bayes
correct = 0
for i in range(0,4999):
    if (ytest[i+15000] == y_pred2[i]):
        correct += 1
acc = correct/5000.00
print "Accuracy of Naive Bayes %f" % acc


# In[64]:

err = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range (0,4999):
    if (ytest[i+15000] != y_pred2[i]):
        err[y_pred2[i] - ytest[i+15000] + 4] += 1

print "Error distribution of naive bayes:"
print err


# In[65]:

bar_width = 1. # set this to whatever you want
data = np.array(err)
positions = np.arange(9)
plt.bar(positions, data, bar_width)
plt.xticks(positions + bar_width / 2, ('-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'))
plt.show()


# In[66]:

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', 
           stop_words='english')
X = vectorizer.fit_transform(clean_test_reviews)
idf_test = X.toarray()
print(idf_test)
print(idf_test.shape)
Y = vectorizer.fit_transform(clean_train_reviews)
idf_train = Y.toarray()
print(idf_train.shape)
print(idf_train)


# In[67]:

# Initialize an empty list to hold the clean reviews
clean_train_reviews_lemma = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_train_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews_lemma.append( review_to_words( xtrainlemma[i] ) )

from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features_lemma = vectorizer.fit_transform(clean_train_reviews_lemma)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features_lemma = train_data_features_lemma.toarray()


# In[68]:

# Initialize an empty list to hold the clean reviews
clean_test_reviews_lemma = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_test_reviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_test_reviews_lemma.append( review_to_words( xtestlemma[i+15000] ) )


# In[69]:

test_data_features_lemma = vectorizer.transform(clean_test_reviews_lemma)
test_data_features_lemma = test_data_features_lemma.toarray()


# In[70]:

print (test_data_features_lemma.shape, train_data_features_lemma.shape)


# In[71]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_lemma = gnb.fit(train_data_features_lemma, ytrainlemma).predict(test_data_features_lemma)


# In[72]:

#The correctness of naive bayes
correct = 0
for i in range(0,4999):
    if (ytest[i+15000] == y_pred_lemma[i]):
        correct += 1
acc = correct/5000.00
print "Accuracy of Naive Bayes after lemma is %f" % acc


# In[73]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#vectorizer = TfidfVectorizer(min_df=1)
#X = vectorizer.fit_transform(clean_train_reviews)
#idf_train = vectorizer.toarray()
#print(idf_train)
transformer = TfidfTransformer(smooth_idf=False)
trainidf_lemma = transformer.fit_transform(train_data_features_lemma)
trainidf_lemma = trainidf_lemma.toarray()
print(trainidf_lemma)
testidf_lemma = transformer.fit_transform(test_data_features_lemma)
testidf_lemma = testidf_lemma.toarray()
print(testidf_lemma)


# In[74]:

gnb = GaussianNB()
y_pred2_lemma = gnb.fit(trainidf_lemma, ytrainlemma).predict(testidf_lemma)


# In[75]:

#The correctness of naive bayes
correct = 0
for i in range(0,4999):
    if (ytest[i+15000] == y_pred2_lemma[i]):
        correct += 1
acc = correct/5000.00
print "Accuracy of Naive Bayes after lemma w/ Tfidfvetorizer is %f" % acc


# In[76]:

from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 1200) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
# Initialize an empty list to hold the clean reviews
clean_test_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange( 0, num_test_reviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_test_reviews.append( review_to_words( xtest[i+15000] ) )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


# In[77]:

allacc = []
x_axis = []

for idx in range (800, 5200, 200):
    x_axis.append(idx)
    
    from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
# Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
    for i in xrange( 0, num_train_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
        clean_train_reviews.append( review_to_words( xtrain[i] ) )
    
    vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = idx) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
    train_data_features = train_data_features.toarray()

# Initialize an empty list to hold the clean reviews
    clean_test_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
    for i in xrange( 0, num_test_reviews):
    # Call our function for each one, and add the result to the list of
    # clean reviews
        clean_test_reviews.append( review_to_words( xtest[i+15000] ) )

# Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    gnb = GaussianNB()
    y_pred_feature = gnb.fit(train_data_features, ytrain).predict(test_data_features)

#The correctness of naive bayes
    correct = 0
    for i in range(0,4999):
        if (ytest[i+15000] == y_pred_feature[i]):
            correct += 1
    acc = correct/5000.00
    allacc.append(acc)
    print "Accuracy of Naive Bayes w/ %d features is %f" % (idx, acc)
    
    err = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range (0,4999):
        if (ytest[i+15000] != y_pred_feature[i]):
            err[y_pred_feature[i] - ytest[i+15000] + 4] += 1

    print "Error distribution of naive bayes:"
    print err
    
    bar_width = 1. # set this to whatever you want
    data = np.array(err)
    positions = np.arange(9)
    plt.bar(positions, data, bar_width)
    plt.xticks(positions + bar_width / 2, ('-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'))
    plt.show()


# In[78]:

print allacc


# In[79]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(x_axis, allacc, marker='o', markersize=3, color="red")

