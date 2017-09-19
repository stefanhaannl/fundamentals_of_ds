import json
import pprint as pp
import pandas as pd
import numpy as np
import re
import datetime

def get_location(boundingBox):
	try:
		for boxCoords in boundingBox:
			longitude = (boxCoords[0][0]+boxCoords[1][0]+boxCoords[2][0]+boxCoords[3][0])/4
			latitude = (boxCoords[0][1]+boxCoords[1][1]+boxCoords[2][1]+boxCoords[3][1])/4
		return (longitude, latitude)
	except:
		return (longitude, latitude)
    
#Extracts the hyperlinks from the tweet's content
def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''
    
def get_textdict(tweet):
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
            wordlist.append(lowerword)
                
    return (hashtaglist,mentionlist,linklist,' '.join(wordlist).lower(),wordlist)
    
def preprocess_dataframe(df):
    # get text_dict
    df['hashtags'], df['mentions'], df['links'], df['text'], df['words'] = zip(*df['text'].apply(get_textdict))
    del df['text']
    
    # get tweet location
    df['longitude'], df['latitude'] = zip(*df['place/bounding_box/coordinates'].apply(get_location))
    del df['place/bounding_box/coordinates']
    
    # get adjusted time and date
    df['datetime'] = (df['timestamp_ms'].apply(int)/ 1e3).apply(datetime.datetime.fromtimestamp)
    del df['timestamp_ms']

    return df

def add_tweet(tweet,relevant_columns_location):
    """
    Converts a json text line to a list entry that only contains our relevant data
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
    Load and returns the dataframe for a given filename (jsons)
    """
    relevant_columns = ['place/bounding_box/coordinates','place/country','timestamp_ms','text','retweeted','user/screen_name']
    relevant_columns_locations = [column.split('/') for column in relevant_columns]
    dflist = []
    with open(filename) as data_file:    
        for i, line in enumerate(data_file):
            dflist.append(add_tweet(line,relevant_columns_locations))
            if (float(i)/1000).is_integer():
                print "Tweet NO "+str(i)+"..."
    df = pd.DataFrame(dflist,columns=relevant_columns)
    del dflist
    return df

    
if __name__ == "__main__":
    df = load_dataframe('C:\Users\ASUS\Documents\geotagged_tweets.jsons')
    #df = preprocess_dataframe(df)
    pp.pprint(df)