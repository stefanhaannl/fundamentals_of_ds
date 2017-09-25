import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.classify.util
from random import shuffle

#classifiers
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from nltk.classify import maxent
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
import ast
import numpy as np
import string
import pprint as pp
import csv
import pandas as pd
import itertools
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

def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title="Confusion matrix voorspelmodel",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix voorspelmodel" )
    else:
        print('Confusion matrix voorspelmodel')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="gold" if cm[i, j] > thresh else "red")

    plt.tight_layout()
    plt.ylabel('Real sentiment')
    plt.xlabel('Predicted sentiment')
    
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
    #rects1[4].set_color('c')
    #rects1[5].set_color('m')
    
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.title('Comparing classifiers on Election twitter dataset')
    plt.xticks(index+bar_width, (accuracy_dict.keys()))
    plt.legend()

    plt.tight_layout()
    plt.show()
 
def plot_graph2(nb_dict, m_dict, sgd_dict, b_dict):
    """
    Plot accuracy results for all possible algorithms
    """
    n_groups= len(nb_dict)
    
    nb_values  = nb_dict.values()
    m_values  = m_dict.values()
    sgd_values  = sgd_dict.values()
    b_values = b_dict.values()
    
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index -1*bar_width, nb_values, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label = 'Naive Bayes')
                     
    rects2 = plt.bar(index , m_values, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label = 'Multinomial NB')
    rects3 = plt.bar(index+bar_width , sgd_values, bar_width,
                     alpha=opacity,
                     color='y',
                     error_kw=error_config,
                     label = 'SGD')
               
    rects3 = plt.bar(index + 2*bar_width, b_values, bar_width,
                     alpha=opacity,
                     color='c',
                     error_kw=error_config,
                     label = 'Bernoulli NB')       
    
    plt.xlabel('K-fold subsets of dataset')
    plt.ylabel('Accuracy')
    plt.title('Comparing classifiers for sentiment analysis')
    mylist = nb_dict.keys()
    plt.xticks(index+bar_width, ([x+1 for x in mylist]))
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
    """
    # Gaussian NB
    print "Gaussian NB"
    classifierGNB = SklearnClassifier(GaussianNB(), sparse=False).train(trainfeats)
    accuracyGNB =  nltk.classify.util.accuracy(classifierGNB, testfeats)
    accuracy_dict['Gaussian NB']= accuracyGNB
    print "Accuracy: ", accuracyGNB
    print ""
    """
    # Multinomial NB
    print "Multinomial NB"
    classifierM = SklearnClassifier(MultinomialNB()).train(trainfeats)
    accuracyM =  nltk.classify.util.accuracy(classifierM, testfeats)
    accuracy_dict['Multinomial NB']= accuracyM
    print "Accuracy: ", accuracyM
    print ""
    
    """
    # Maximument
    print "Maximum entropy"
    classifierME =  maxent.MaxentClassifier.train(trainfeats, max_iter =20)
    accuracyME =  nltk.classify.util.accuracy(classifierME, testfeats)
    accuracy_dict['MaxEnt']= accuracyME
    print "Accuracy: ", accuracyME
    print ""
    """
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
 
       
def add_sentiment(df, trainfeats):
    # train model
    classifierNB =NaiveBayesClassifier.train(trainfeats)
    classifierSGD =SklearnClassifier(SGDClassifier()).train(trainfeats)
    
    
    # init sentiment predictions
    sentiment_predictions = []
    
    # For each row of our tweet dataframe
    i=0
    for i in range(i,len(df)):
        
        # get tweet
        wordlist = df['words'].iloc[i]
        #wordlist = ['I', 'hate', 'you' ]
        
        # predict sentiment
        predictionNB = classifierNB.classify(word_feats(wordlist))
        predictionSGD = classifierSGD.classify(word_feats(wordlist))
        if (predictionNB == 'pos') and (predictionSGD == 'pos'):
            prediction = 1
            
        elif (predictionNB == 'neg') and (predictionSGD == 'neg'):
            prediction = 0
        
        elif (predictionNB == 'neg') and (predictionSGD == 'pos'):
            prediction = 1
        elif (predictionNB == 'pos') and (predictionSGD == 'neg'):
            prediction = 0
        
        else:
            print "Error"
            
        # add sentiment to list
        sentiment_predictions.append(prediction)
        
    #add list as sentiment column dataframe
    df['sentiment'] = sentiment_predictions
    
    return df
    
 
def k_test(kfolds, dataset):
    num_folds = kfolds
    subset_size = len(dataset)/num_folds
    nb_dict = {}
    m_dict ={}
    sgd_dict ={}
    b_dict = {}
    
    for i in range(num_folds):
        testing_this_round = dataset[i*subset_size:][:subset_size]
        training_this_round = dataset[:i*subset_size] + dataset[(i+1)*subset_size:]
        
        # train using training_this_round
        classifierNB = NaiveBayesClassifier.train(training_this_round)
        classifierM = SklearnClassifier(MultinomialNB()).train(training_this_round)
        classifierSGD = SklearnClassifier(SGDClassifier()).train(training_this_round)
        classifierB = SklearnClassifier(BernoulliNB()).train(training_this_round)
        
        # evaluate against testing_this_round
        accuracyNB =  nltk.classify.util.accuracy(classifierNB, testing_this_round)
        accuracyM =  nltk.classify.util.accuracy(classifierM, testing_this_round)
        accuracySGD =  nltk.classify.util.accuracy(classifierSGD, testing_this_round)
        accuracyB =  nltk.classify.util.accuracy(classifierB, testing_this_round)
    
        # save accuracy
        nb_dict[i] = accuracyNB
        m_dict[i]  = accuracyM
        sgd_dict[i] = accuracySGD
        b_dict[i] = accuracyB
        
    return nb_dict, m_dict, sgd_dict, b_dict
   
if __name__ == "__main__":

    ##### Train and test on twitter database ####
    """
    # create datasbase of negative and positive tweets
    datafile = 'Sentiment Analysis Dataset.csv'
    used_tweets = 10000
    positiveTweets, negativeTweets = createDatabase(used_tweets, datafile)
    trainfeats = positiveTweets + negativeTweets
    shuffle(trainfeats)
    """
    
    trainfeats = pd.read_pickle('train.pkl')
    df = pd.read_excel('sentiment_tweets.xlsx')
    words = df.words
    
    df = add_sentiment(df, trainfeats.tuples)
    
    y_pred = df.sentiment
    y_test = df.hand_classified
    accuracy = (accuracy_score(y_test, y_pred))*100
    #accuracy_dict = {}
    #accuracy_dict['Naive Bayes'] = 0.62
    #accuracy_dict['Multinomial NB'] = 0.56
    #accuracy_dict['SGD'] = 0.62
    #accuracy_dict['Bernoulli NB'] = 0.60
    
    #plot_graph(accuracy_dict)
    
    plt.figure()
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=[ 'Negative tweets','Positive tweets'], normalize=False,
                      title='Confusion matrix Naive Bayes')

    plt.show()
    
    #nb_dict, m_dict, sgd_dict, b_dict = k_test(4, trainfeats)
    #plot_graph2(nb_dict, m_dict,sgd_dict, b_dict)
    
    
    # split train en test set
    #train_proportion = float(7)/float(8)
    #trainfeats, testfeats = create_train_test(positiveTweets, negativeTweets, train_proportion)
    
    # make pickle dataframe
    #train_df = pd.DataFrame()
    #train_df['tuples'] = trainfeats
    #train_df.to_pickle("train1mil.pkl")
    
    """
    # read pickle ; only use for testing
    #df = pd.read_pickle("train.pkl")
    #train_list = df.tuples.tolist()
  
    #classifierNB = NaiveBayesClassifier.train(tuple_list)
    #prediction = classifierNB.classify(word_feats(wordlist))
    #pp.pprint(prediction)
    """ 
   
    #print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    # compare algorithms to choose best model (NAive Bayes best -> see graph)
    #compare_algorithms(trainfeats, testfeats)
    
    
