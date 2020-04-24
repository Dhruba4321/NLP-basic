import requests
from bs4 import BeautifulSoup
import pickle
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

# Scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
    page = requests.get(url).text # Get all data from URL
    soup = BeautifulSoup(page, "lxml") # Read as an HTML document
    text = [p.text for p in soup.find(class_="post-content").find_all('p')] # Pull out all text from post-content
    print(url)
    return text

# URLs of transcripts in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']
# Comedian names
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']

# Actually request transcripts
#transcripts = [url_to_transcript(u) for u in urls]

# Pickle (save) files for later use
# We pickle because we want to avoid performing API calls on site multiple times
# create a folder called transcripts from the terminal by running the command- mkdir transcripts

#for i, c in enumerate(comedians):
    #with open("transcripts/" + c + ".txt", "wb") as file:
        #pickle.dump(transcripts[i], file)

# Load pickled files
data = {}
for i, c in enumerate(comedians):
    with open("transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)

#Looking at the data
#print(data.keys())
#print(data['louis'][:2])

# The dictionary is currently in the format of
# key: comedian, value: list of text
# print(next(iter(data.values())))

# We are going to change this one giant string of text.
# New format will be {key: comedian, value: string}
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text
# Combining it!
data_combined = dict()
for key, value in data.items():
    data_combined[key] = combine_text(value)

# Converting from dictionary format to DataFrame
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame(data_combined, index=[0]).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()

# Print the Corpus
# print(data_df)
# print(data_df.columns)
# Let's take a look at the transcript for Ali Wong
# print(data_df.transcript.loc['ali'])

# Apply a first round of text cleaning techniques
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
# print(data_clean)

# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional quotation marks and newline text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)
# Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
# print(data_clean)
data_df.to_pickle("corpus.pkl")
# create a document-term matrix using CountVectorizer,
# and exclude common English stop words
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
# Convert it to an array and label all the columns
# Can use this part for future projects
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

# Let's also pickle the cleaned data
# (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))

#Document-Term Matrix
# print(data_dtm)

# Let's pickle it for later use
data_dtm.to_pickle("dtm.pkl")
