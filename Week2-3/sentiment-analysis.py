import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.classify.util

#classifiers
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from nltk.classify import maxent

import numpy as np
import string
import pprint as pp
import csv

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

# Create a set of stopwords
stop = set(stopwords.words('english'))

# Create a set of punctuation words 
exclude = set(string.punctuation)

# Make sure it are only ascii letter
printable = set(string.printable)

# This is the function makeing the lemmatization
lemma = WordNetLemmatizer()

def plot_graph(accuracy_dict):
    """
    Plot accuracy results for all possible algorithms
    """
    n_groups= len(accuracy_dict)
    
    accuracy_values  = accuracy_dict.values()
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.55
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index+ 0.5*bar_width, accuracy_values, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config)
    rects1[0].set_color('y')
    rects1[1].set_color('r')
    rects1[2].set_color('b')
    rects1[3].set_color('g')
    rects1[4].set_color('c')
    
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.title('Comparing classifiers for sentiment analysis')
    plt.xticks(index+bar_width, (accuracy_dict.keys()))
    plt.legend()

    plt.tight_layout()
    plt.show()
 
def compare_algorithms(trainfeats, testfeats):
    accuracy_dict = {}
    
    # NaiveBayesClassifier
    print "Naive Bayes"
    classifierNB = NaiveBayesClassifier.train(trainfeats)
    accuracyNB =  nltk.classify.util.accuracy(classifierNB, testfeats)
    accuracy_dict['NaiveBayes']= accuracyNB
    classifierNB.show_most_informative_features()
    print "Accuracy: ", accuracyNB
    print ""

       
    # BernoulliNBClassifier
    print "Bernoulli NB"
    classifierB = SklearnClassifier(BernoulliNB()).train(trainfeats)
    accuracyB =  nltk.classify.util.accuracy(classifierB, testfeats)
    accuracy_dict['Bernoulli NB']= accuracyB
    print "Accuracy: ", accuracyB
    print ""
    
     # SGD
    print "SGD"
    classifierSGD = SklearnClassifier(SGDClassifier()).train(trainfeats)
    accuracySGD =  nltk.classify.util.accuracy(classifierSGD, testfeats)
    accuracy_dict['SGD']= accuracySGD
    print "Accuracy: ", accuracySGD
    print ""
    
    # Gaussian NB
    print "Gaussian NB"
    classifierGNB = SklearnClassifier(GaussianNB(), sparse=False).train(trainfeats)
    accuracyGNB =  nltk.classify.util.accuracy(classifierGNB, testfeats)
    accuracy_dict['Gaussian NB']= accuracyGNB
    print "Accuracy: ", accuracyGNB
    print ""
    
    # Multinomial NB
    print "Multinomial NB"
    classifierM = SklearnClassifier(MultinomialNB()).train(trainfeats)
    accuracyM =  nltk.classify.util.accuracy(classifierM, testfeats)
    accuracy_dict['Multinomial NB']= accuracyM
    print "Accuracy: ", accuracyM
    print ""
    
    plot_graph(accuracy_dict)

# In this function we perform the entire cleaning
def clean(doc):
    #stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    #punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    punc_free = ''.join(ch for ch in doc if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def createWordlist(sentence):
    sentence = filter(lambda x: x in printable, sentence) # Remove non-ascii letters
    token = nltk.word_tokenize(sentence)
    doc_clean = [clean(doc).split() for doc in token]
    allWords = []
    for word in doc_clean:
        if len(word) > 0:
            allWords.append(unicode(word[0]))   # Turn it into unicode     
    return allWords

def word_feats(words): #Create a dictionary
    return dict([(word, True) for word in words])

def createDatabase(amountInDatabase, datafile): #Create the test set with a size of amountInDatabase
    positiveTweets = []
    negativeTweets = []
    with open(datafile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row, i  in zip(reader, range(0,amountInDatabase)):
            print (i/float(amountInDatabase))
            allWords = createWordlist(row["SentimentText"])
            if int(row["Sentiment"]) is 0:
                negativeTweets.append((word_feats(allWords), 'neg'))
            else: 
                positiveTweets.append((word_feats(allWords), 'pos'))
    return positiveTweets, negativeTweets

def create_train_test(positiveTweets, negativeTweets, train_proportion):
    print len(negativeTweets)
    print train_proportion
    negcutoff = int(len(negativeTweets)*train_proportion)
    print negcutoff
    poscutoff = int(len(positiveTweets)*train_proportion)
    print poscutoff
     
    # Construct the training dataset containing 50% positive reviews and 50% negative reviews
    trainfeats = negativeTweets[:negcutoff] + positiveTweets[:poscutoff]

    # Construct the test dataset containing 50% positive reviews and 50% negative reviews
    testfeats = negativeTweets[negcutoff:] + positiveTweets[poscutoff:]
    
    return trainfeats, testfeats
    
def add_sentiment(df):
    # TO DO: script that adds sentiment column to dataframe 
    # STEPS: 
    # train model
    classifierNB = NaiveBayesClassifier.train(trainfeats)
    
    # init sentiment predictions
    sentiment_predictions = []
    
    # For each row of our tweet dataframe
    i=0
    for i in range(i,len(df)):
        
        # get tweet
        wordlist = df['words'].iloc[i]
        #wordlist = ['I', 'hate', 'you' ]
        
        # predict sentiment
        prediction = classifierNB.classify(word_feats(wordlist))
        
        # add sentiment to list
        sentiment_predictions.append(prediction)
        
    #add list as sentiment column dataframe
    df['sentiment'] = sentiment_predictions
    
    return df
    
    
if __name__ == "__main__":

    ##### Train and test on twitter database ####
    
    # create datasbase of negative and positive tweets
    datafile = 'Sentiment Analysis Dataset.csv'
    used_tweets = 10000
    positiveTweets, negativeTweets = createDatabase(used_tweets, datafile)

    # split train en test set
    train_proportion = float(3)/float(4)
    trainfeats, testfeats = create_train_test(positiveTweets, negativeTweets, train_proportion)
    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    # compare algorithms to choose best model (NAive Bayes best -> see graph)
    compare_algorithms(trainfeats, testfeats)
    
    # adds sentiment column to dataframe
    """
    df = preprocessing
    df = add_sentiment(df)
    """
