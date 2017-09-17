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
    
def get_location(text):
    pass
    
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
    
def preprocess_dataframe(df):
    # get text_dict
    df['textinfo'] = df['text'].apply(get_textdict)
    
    # alleen nog:
    """
    - lat/long
    - Adjusted time
    
    """
    
    # fill dataframe    
    df['longitude'] = 0.2
    df['latitude'] = 0.3
    df['date'] = (df['timestamp_ms'].apply(int)/ 1e3).apply(datetime.datetime.fromtimestamp)
    df['new-york time'] = 0.3
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
        else:
            d[column] = struct[c[0]][c[1]]
    return d

    
if __name__ == "__main__":
    # Read only the relevant data
    filename = 'tweets_sample.jsons'
    relevant_columns = ['place/country','timestamp_ms','text','retweeted','user/screen_name']
    df = pd.DataFrame(columns=relevant_columns)
    with open(filename) as data_file:    
        for i, line in enumerate(data_file):
            df.loc[i] = tweet_to_dict(line,relevant_columns)
    
    # Preprocess the data into our ideal dataframe
    df = preprocess_dataframe(df)
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
    
    
    
    
