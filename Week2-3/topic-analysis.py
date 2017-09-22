import numpy as np
import pandas as pd
import string
import pickle
import gensim
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora,models,similarities

def clean_words(doc):
    """
    Remove stopwords from string and normalize words in string
    INPUT: Series of wordlists(which are the documents)
    OUTPUT: Corpora dictionary of list of words
    """
    doc = list(doc)
    result_list = []
    # Create a set of stopwords
    stop = set(stopwords.words('english'))
    # This is the function makeing the lemmatization
    lemma = WordNetLemmatizer()
    for lst in doc:
        stop_free = " ".join([i for i in lst if i not in stop])
        normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        result_list.append(normalized.split())
    return result_list

def load_tweets(filepath):
    """
    Load an array of tweets two work with. (pickle file)
    INPUT: filepath to your pickle file
    OUTPUT: Pandas series of tweet documents (as lists)
    """
    df = pd.read_pickle(filepath)
    return df[['words','hashtags']]

def cluster_by_hashtags(df,n):
    """
    Generates a corpus based on the n'th most prevalent hashtags
    INPUT: Dataframe with hashtags (list) and text (list) column
    OUTPUT: Dataframe with hashtag (str) and text(list) column
    """
    result_df = []
    hashtags = []
    for lst in df['hashtags']:
        hashtags.extend(lst)
    hashtags = pd.Series(hashtags)
    hashtag_values = hashtags.value_counts()[0:n]
    hashtags_list = hashtag_values.index.values.tolist()
    for hashtag in hashtags_list:
        print "Filtering words for hashtag: "+hashtag
        words = []
        for row in df.itertuples():
            row = tuple(row)
            if hashtag in row[2]:
                words.extend(row[1])
        result_df.append([hashtag, words])
    return pd.DataFrame(result_df,columns=['hashtag','words'])

def testDoc_loadingpart(filepath):
    # Load datafile
    print "Loading datafile..."
    df = load_tweets(r'C:\Users\shaan\Documents\true_tweets.pkl')
    
    # Clean words
    print "Cleaning words..."
    doc_clean = clean_words(df['words'])
    
    return doc_clean

def testDoc_testingpart(doc_clean, doc_n = 1000, topic_n = 5, iter_n = 100):
    doc_clean = doc_clean[0:doc_n]
    
    # Creating a corpora dictionary from the cleaned document
    print "Generating acorpora dictionary..."
    dictionary = corpora.Dictionary(doc_clean)
    
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.   
    print "Convert list of documents into a matrix..."
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    
    # Creating the object for LDA model using gensim library
    print "Creating an LDA model..."
    Lda = gensim.models.ldamodel.LdaModel
    
    # Running and Trainign LDA model on the document term matrix.
    print "Run and train the LDA model on the matrix..."
    ldamodel = Lda(doc_term_matrix, num_topics=topic_n, id2word = dictionary, passes=iter_n)
    
    return ldamodel

def printTopics(ldamodel, maxNumberofTopics, wordsPerTopic):
    """  
    prints the consturcted topics
    Input: ldamodel, maximum numbers of topics you want to be printed, 
        number of words you want printed
    """
    topics = ldamodel.print_topics(num_topics=3, num_words=8)
    i=0
    for topic in topics:
        print "Topic",i ,"->", topic
        i+=1

def topicNewDoc(ldamodel, doc):
    vec_bow = dictionary.doc2bow(doc.lower().split())
    return(ldamodel[vec_bow])


