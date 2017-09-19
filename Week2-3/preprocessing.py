import json
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
		return (np.nan, np.nan)
    
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
    print "Starting the preprocessing!"
    
    # get text_dict
    print "Extracting the hashtags, mentions, links, text, and words..."
    df['hashtags'], df['mentions'], df['links'], df['text'], df['words'] = zip(*df['text'].apply(get_textdict))
    del df['text']
    
    # get tweet location
    print "Extracting the longitude and latitude from the location..."
    df['longitude'], df['latitude'] = zip(*df['coordinates'].apply(get_location))
    del df['coordinates']
    
    # get adjusted time and date
    print "Converting the timestamp to a datetime format..."
    df['datetime'] = (df['timestamp_ms'].apply(int)/ 1e3).apply(datetime.datetime.fromtimestamp)
    del df['timestamp_ms']
    
    # remove the tweets without words
    df = df[df['words'].apply(lambda x: x != [])]
    
    print "Finished the preprocessing!"
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
    print "Start loading the dataframe!"
    dflist = []
    with open(filename) as data_file:    
        for i, line in enumerate(data_file):
            #filter the tweets on US tweets
            tweet = add_tweet(line,relevant_columns_locations)
            if (tweet[1] == 'United States') and (np.nan not in tweet)
                dflist.append(tweet)
            if (float(i)/10000).is_integer():
                print "Tweet NO "+str(i)+"..."
    df = pd.DataFrame(dflist,columns=[column[-1] for column in relevant_columns_locations])
    del dflist
    print "Finished loading the dataframe!"
    return df

    
if __name__ == "__main__":
    df = load_dataframe('C:\Users\shaan\Documents\geotagged_tweets.jsons')
    df = preprocess_dataframe(df)
    print df.head()