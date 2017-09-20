import json
import pandas as pd
import numpy as np
import re
import datetime
import pprint as pp
from nltk import tokenize
from nltk.classify import NaiveBayesClassifier

def get_location(boundingBox):
    """
    Extracts the boxcoords from a boundingbox value to a longitude and latitude. Returns NaN if the value is not accessible.
    INPUT: list BoundingBox
    OUTPUT: tuple of longitude and latitude
    """
    try:
        for boxCoords in boundingBox:
            longitude = (boxCoords[0][0]+boxCoords[1][0]+boxCoords[2][0]+boxCoords[3][0])/4
            latitude = (boxCoords[0][1]+boxCoords[1][1]+boxCoords[2][1]+boxCoords[3][1])/4
        return (longitude, latitude)
    except:
        return (np.nan, np.nan)
    
def extract_link(text):
    """
    Extract links from at text using reges
    INPUT: text
    OUTPUT: list of links
    """
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''
    
def get_textdict(tweet):
    """
    Converts the text of a tweet to a list of hashtags, mentions, links and words.
    INPUT: string of tweet text
    OUTPUT: tuple of lists.
    """
    # divide words
    tweet = tweet.encode('ascii','ignore')
    textlist = str(tweet).split()
    hashtaglist=[]
    mentionlist = []
    linklist = []
    wordlist = []
    
    # loop through all words
    for word in textlist:
        # separate charachters
        charachters = list(word)
        
        # hashtag
        if  charachters[0] == '#':
            hashtaglist.append(word)
        # mention
        elif  charachters[0] == '@':
            mentionlist.append(word)
        # link
        elif  word[:4] == 'http':
            linklist.append(word)
        # normal word    
        else:
            lowerword = word.lower()
            cleaned_lowerword = tokenize.RegexpTokenizer(r'\w+').tokenize(lowerword)
            for word in cleaned_lowerword:
                wordlist.append(str(word))
                
    return (hashtaglist,mentionlist,linklist,' '.join(wordlist).lower(),wordlist)
    
def preprocess_dataframe(df):
    """
    Preprocesses the dataframe. Extracts the hashtags, mentions, links, text and words from the text. Longitude and latitude from the location. Adjusted timedat. Filter on tweets which only contains words.
    INPUT: Pandas Dataframe
    OUTPUT: Pandas Dataframe
    """
    print "Starting the preprocessing!"
    
    # get text_dict
    print "Extracting the hashtags, mentions, links, text, and words..."
    df['hashtags'], df['mentions'], df['links'], df['text'], df['words'] = zip(*df['text'].apply(get_textdict))
    del df['text']
    
    # get tweet location
    if 'coordinates' in df.columns:        
        print "Extracting the longitude and latitude from the location..."
        df['longitude'], df['latitude'] = zip(*df['coordinates'].apply(get_location))
        del df['coordinates']
    
    # get adjusted time and date
    print "Converting the timestamp to a datetime format..."
    if 'timestamp_ms' in df.columns:
        df['datetime'] = (df['timestamp_ms'].apply(int)/ 1e3).apply(datetime.datetime.fromtimestamp)
        del df['timestamp_ms']
    elif 'created_at' in df.columns:
        df['datetime'] = df['created_at'].apply(lambda x: datetime.strptime(x, '%m-%d-%Y %H:%M:%S'))
        del df['created_at']
        
    # remove the tweets without words
    print "Filtering out empty tweets..."
    df = df[df['words'].apply(lambda x: x != [])]
    
    print "Finished the preprocessing!"
    return df

def add_tweet(tweet,relevant_columns_location):
    """
    Converts a json text line to a list entry that only contains our relevant data. Filteres out the rest of the tweet data.
    INPUT: str: tweet, list: relevant_columns_location (for nested information)
    OUTPUT: list of relevant columns extracted
    """
    struct = json.loads(tweet)
    d = []
    for c in relevant_columns_location:
        try:
            if len(c) == 1:
                d.append(struct[c[0]])
            elif len(c) == 2:
                d.append(struct[c[0]][c[1]])
            else:
                d.append(struct[c[0]][c[1]][c[2]])
        except:
            d.append(np.nan)
            print "Added NAN value"
    return d

def load_dataframe(filename):
    """
    Load and returns the dataframe for a given filename (jsons). Relevant columns are specified in the function.
    INPUT: Filepath
    OUTPUT: Pandas dataframe of relevant columns
    """
    relevant_columns = ['place/bounding_box/coordinates','place/country','timestamp_ms','text','retweeted','user/screen_name']
    relevant_columns_locations = [column.split('/') for column in relevant_columns]
    print "Start loading the dataframe!"
    dflist = []
    with open(filename) as data_file:    
        for i, line in enumerate(data_file):
            #filter the tweets on US tweets
            tweet = add_tweet(line,relevant_columns_locations)
            if (tweet[1] == 'United States') and (np.nan not in tweet):
                dflist.append(tweet)
            if (float(i)/10000).is_integer():
                print "Tweet NO "+str(i)+"..."
    df = pd.DataFrame(dflist,columns=[column[-1] for column in relevant_columns_locations])
    del dflist
    print "Finished loading the dataframe!"
    return df

def load_pandas():
    """
    Reads a .pkl file for a dataframe format. Specifiy the path in the function.
    OUTPUT: Pandas Dataframe
    """
    pandasfilepath = r'C:\Users\shaan\Documents\true_tweets.pkl'
    return pd.read_pickle(pandasfilepath)
    

def create_wordseries(df_series):
    """
    Converts a pandas series of of lists to a pandas series of all values in the combined lists. Can be used on the words column of a dataset to acquire a series of all the used words.
    INPUT: Pandas Series
    OUTPUT: Pandas Series
    """
    lst = []
    for wordlist in df_series:
        lst.extend(wordlist)
    return pd.Series(lst)

def load_trumptweets(path):
    print "Loading the trump csv file..."
    df = pd.read_csv(path, usecols = ['text','created_at','retweet_count','favorite_count','is_retweet'])
    return df
def word_feats(words): 
    #Create a dictionary
    return dict([(word, True) for word in words])
    
def add_sentiment(df, trainfeats):
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
        if prediction == 'pos':
            prediction = 1
        else:
            prediction = 0
        
        # add sentiment to list
        sentiment_predictions.append(prediction)
        
    #add list as sentiment column dataframe
    df['sentiment'] = sentiment_predictions
    
    return df
    
if __name__ == "__main__":
    #df = load_pandas()
    #df_trump = load_trumptweets('data/trumptweets.csv')
    #df = preprocess_dataframe(df)
    #print df.head()
    
    
    df = load_dataframe('tweets_sample.jsons')
    df = preprocess_dataframe(df)
    
    # get trainingset
    trainpickle = pd.read_pickle("train.pkl")
    trainset = trainpickle.tuples.tolist()
    
    # add sentument
    df = add_sentiment(df, trainset)
    pp.pprint(df)
    
    #print df.head()
    pass
