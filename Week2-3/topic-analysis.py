import numpy as np
import pandas as pd
import string
import pickle
import gensim
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora,models,similarities

def clean(doc):
    """
    Remove stopwords from string and normalize words in string
    INPUT: List of strings(which are the documents)
    OUTPUT: List of list of words
    """
    
    # Create a set of stopwords
    stop = set(stopwords.words('english'))

    # This is the function makeing the lemmatization
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    return normalized

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

def testDoc():
    doc1 = "Axel is the hero of this Data Science Group"
    doc2 = "Information Studies Data Science is awsome"
    doc3 = "Do you know Axel he is a hero in Data Science Daniel"
    doc4 = "Daniel exist which is probably a good thing"
    doc5 = "Daniel oh Daniel are you my lover"

    # compile documents
    doc_complete = [doc1, doc2, doc3, doc4, doc5]
    doc_clean = [clean(doc).split() for doc in doc_complete]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=100)
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


