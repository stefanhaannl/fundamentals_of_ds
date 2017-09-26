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
    stop.add('amp')
    stop.add('u')
    stop.add('2')
    stop.add('4')
    stop.add('rt')
    stop.add('w')
    stop.add('r')
    stop.add('c')
    stop.add('said')
    stop.add('says')
    stop.add('0')
    stop.add('ky')
    stop.add('wh')
    stop.add('tko')
    stop.add('gt')
    stop.add('yep')
    stop.add('40')
    stop.add('hc')
    stop.add('fr')
    stop.add('n')
    stop.add('sgspts')
    stop.add('7')
    stop.add('25')
    stop.add('50')
    stop.add('de')
    stop.add('bo')
    stop.add('ag')
    stop.add('b')
    stop.add('1')
    stop.add('n')
    stop.add('co')
    stop.add('cglbr')
    stop.add('17ml')
    stop.add('wh')
    stop.add('dx')
    stop.add('19')
    stop.add('v')
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
        words = []
        for row in df.itertuples():
            row = tuple(row)
            if hashtag in row[2]:
                words.extend(row[1])
        result_df.append([hashtag, words])
    return pd.DataFrame(result_df,columns=['hashtag','words'])

def testDoc_loadingpart(filepath,amountHashtags = 0):
    # Load datafile
    print "Loading datafile..."
    df = load_tweets(filepath)
    if amountHashtags != 0:
        print "Clustering hashtags..."
        df = cluster_by_hashtags(df,amountHashtags)
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
    
    printTopics(ldamodel,10,10)
    
    return dictionary, ldamodel

def testDoc_parameters(filepath, hashtag_n, topic_n, iter_n, outputfile):
    """
    Starts an iteration for testing the LDA model on multiple parameters.
    INPUT filepath: str, the filepath to the pickle file
    INPUT hashtag_n: list, list of numers of hashtags you want to test
    INPUT topic_n: list, list of numers of topics you want to test
    INPUT: iter_n: int, numer of iterations for the model
    INPUT: outputfile: str, the filepath of where to save the lda models.
    This function saves every lda model you tested and prints the topics to screen for review.
    """
    df = testDoc_loadingpart(filepath,max(hashtag_n))
    print "Starting testing phase:"
    print "Hashtags"
    print hashtag_n
    print "Topics"
    print topic_n
    for hashtag in hashtag_n:
        for topic in topic_n:
            print "Simulation for hashtag_n: "+str(hashtag)+" and topic_n: "+str(topic)
            print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            lda = testDoc_testingpart(df,hashtag,topic,iter_n)
            lda.save(filepath+"lda_hashtags"+str(hashtag)+"_topics"+str(topic)+".model")
            print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

def printTopics(ldamodel, maxNumberofTopics, wordsPerTopic):
    """  
    prints the consturcted topics
    Input: ldamodel, maximum numbers of topics you want to be printed, 
        number of words you want printed
    """
    topics = ldamodel.print_topics(num_topics=maxNumberofTopics, num_words=wordsPerTopic)
    i=0
    for topic in topics:
        print "Topic",i ,"->", topic
        i+=1

def topicNewDoc(doc):
    """
    Need to have dictionary and ldamodel globally defined
    """
    vec_bow = dictionary.doc2bow(doc)
    return(ldamodel[vec_bow])

def calculate_topic_for_df(picklefilepath, n = 0, reverse = False):   
    """
    Add a topic_vector from a tweet for a whole dataframe. Do not forget to assign to a variable since the function returns your new dataframe
    INPUT: picklefilepath: str, path to your pickle file
    INPUT: n = 0: int, numer of rows you want apply it on (n = 0 means all)
    INPUT: reverse: bool, set True if you want to start at the end
    """
    print "Loading the pickle file..."
    df = pd.read_pickle(picklefilepath)
    if reverse == True:
        df.reindex(index=df.index[::-1])
    if n != 0:
        df = df.head(n)
    print "Loading the lda model..."
    global ldamodel
    ldamodel = models.LdaModel.load(r"ldamodel/ldamodel.model")
    global dictionary
    dictionary = corpora.Dictionary.load(r"ldamodel/ldadict")
    print "Applying the model..."
    df['topic_v'] = df['words'].apply(topicNewDoc)
    return df