import json
import pprint as pp
import pandas as pd
import re
import datetime

#Removes HTML tags in a set of data
def RemoveHTMLTags(data):
    p = re.compile(r'<[^<]*?>')
    return p.sub('', data)
    
#Checks whether a word is included in the tweet's content
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False
    
def get_location(boundingBox):
    i = 0
    for i in range(i,len(boundingBox)):
        boxCoords = boundingBox[i]
        longitude = (boxCoords[0][0]+boxCoords[1][0]+boxCoords[2][0]+boxCoords[3][0])/4
        latitude = (boxCoords[0][1]+boxCoords[1][1]+boxCoords[2][1]+boxCoords[3][1])/4
    return (longitude, latitude)
    
def get_adjusted_datetime(text):
    pass 
    
#Extracts the hyperlinks from the tweet's content
def extract_link(text):
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(regex, text)
    if match:
        return match.group()
    return ''
    
def get_textdict(tweet):
    # divide words
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

def tweet_to_dict(tweet,relevant_columns):
    """
    Converts a json text line to a dictionary that only contains our relevant data
    """
    struct = json.loads(tweet)
    d = {}
    for column in relevant_columns:
        c = column.split('/')
        if len(c) == 1:
            d[column] = struct[c[0]]
        elif len(c) == 2:
            d[column] = struct[c[0]][c[1]]
        else:
            d[column] = struct[c[0]][c[1]][c[2]]
    return d

    
if __name__ == "__main__":
    # Read only the relevant data
    filename = 'tweets_sample.jsons'
    relevant_columns = ['place/bounding_box/coordinates','place/country','timestamp_ms','text','retweeted','user/screen_name']
    df = pd.DataFrame(columns=relevant_columns)
    with open(filename) as data_file:    
        for i, line in enumerate(data_file):
            df.loc[i] = tweet_to_dict(line,relevant_columns)
    
    # Preprocess the data into our ideal dataframe
    df = preprocess_dataframe(df)
    pp.pprint(df)