# We need to import json for dumping the tweets into our file
import json 
import pprint as pp
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
import HTMLParser # In Python 3.4+ import html 
import nltk
import datetime

def RemoveHTMLTags(data):
    p = re.compile(r'<[^<]*?>')
    return p.sub('', data)
    

    
# A function that checks whether a word is included in the tweet's content
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False
    
#def get_location(text):
    # daniel
#def get_adjusted_datetime(text):
    # daniel  
    
      
# A function that extracts the hyperlinks from the tweet's content.
def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''
    
def get_textdict(tweet):
    # divide words
    textlist = str(tweet['text']).split()
    
    # init    text_dict = {}
    hashtaglist=[]
    mentionlist = []
    linklist = []
    wordlist = []
    
    # loop through all words
    i=0
    for i in range(i, len(textlist)):
        word = textlist[i]
        
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
            wordlist.append(lowerword)
       
    # make text_dict
    text_dict = {'hashtags':hashtaglist,
                'mentions':mentionlist,
                'links':linklist,
                'text':' '.join(wordlist).lower(),
                'wordlist': wordlist,
                }
                
    return text_dict
    
def get_dict(tweet):
    # get text_dict
    text_dict = get_textdict(tweet)
    # alleen nog:
    """
    - lat/long
    - Adjusted time
    
    """
    # fill dataframe    
    tweet_dict = {}
    tweet_dict['longitude'] = 0.2
    tweet_dict['latitude'] = 0.3
    tweet_dict['country'] = tweet['place']['country']
    tweet_dict['date'] =  datetime.datetime.fromtimestamp(int(tweet['timestamp_ms'])/ 1e3)
    tweet_dict['new-york time'] = 0.3
    tweet_dict['text']=text_dict['text']
    tweet_dict['mentions'] = text_dict['mentions']
    tweet_dict['hashtags'] = text_dict['hashtags']
    tweet_dict['links'] =  text_dict['links']
    tweet_dict['retweet'] = tweet['retweeted']
    tweet_dict['wordlist'] = text_dict['wordlist']
    tweet_dict['user'] = tweet['user']['screen_name']
    return tweet_dict
    
def make_dataframe(total_tweets):
    
    length_dataset = len(total_tweets)
    total_dict = {}
    i=0
    # loop through all tweets
    for i in range(i,length_dataset):
        # single tweet
        tweet = total_tweets[i]
        
        # get dict with tweet info
        tweet_dict = get_dict(tweet)
        
        # add to toal_dict with index i
        total_dict[i] = tweet_dict
    
    # convert dict to dataframe
    dataframe = pd.DataFrame.from_dict(total_dict, orient='index')
    
    return dataframe
    
    
if __name__ == "__main__":
    #tweets = pd.read_json('tweets_sample.jsons' )
    
    # read jsons in file
    total_tweets = []
    i=0
    with open('tweets_sample.jsons') as data_file:    
        for line in data_file:
            total_tweets.append(json.loads(line))
    print "Total tweets: ", len(total_tweets)
    
    # make dataframe / preprocessing
    df = make_dataframe(total_tweets)
    pp.pprint(df)
    
    
    # add column with sentiment of each tweet
    #df_with_sentiment = add_sentiment(df)
    
    
    # overbodige comments
    # coordinates
    # created_at
    # place / boundingbox / coordinates (midden)
    # geo
    # text (mentions, text, plaatjes link)
    # time zone
    
    
    
    
