import pandas as pd
from collections import Counter
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# Read in the document-term matrix
data = pd.read_pickle('dtm.pkl')
#Transpose becuse it's harder to operate across rows. Easier across columns.
#We want to aggregate for each comedian. So comedians should be on the columns.
data = data.transpose()
# print(data.head())
# Finding the top 30 words said by each comedian
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

# print(top_dict)
# Print the top 15 words said by each comedian
for comedian, top_words in top_dict.items():
    print(comedian)
    print(', '.join([word for word, count in top_words[0:15]]))
    print('---')
# Look at the most common top words --> add them to the stop word list

# Let's first create a list that just has each comedians top 30 words (even if repeated)
words = []
for comedian in data.columns:
    top = [word for (word, count) in top_dict[comedian]]
    for t in top:
        words.append(t)

# print(words)
# Aggregate this list and identify the most common words
# along with how many comedian's routines they occur in
print(Counter(words).most_common())
# If more than half of the comedians have it as a top word,
# exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
print(add_stop_words)

# Read in cleaned data from corpus
data_clean = pd.read_pickle('data_clean.pkl')

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix which excludes our additional stop words
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")

# Making word-clouds
wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [16, 6]

full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

# Create subplots for each comedian
# for index, comedian in enumerate(data.columns):
    # wc.generate(data_clean.transcript[comedian])

    # plt.subplot(3, 4, index + 1)
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.title(full_names[index])

# plt.show()

# Find the number of unique words that each comedian uses

# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for comedian in data.columns:
    uniques = data[comedian].nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
print(data_unique_sort)

# Calculate the words per minute of each comedian

# Find the total number of words that a comedian uses
total_list = []
for comedian in data.columns:
    totals = sum(data[comedian])
    total_list.append(totals)

# Comedy special run times from IMDB (in minutes)
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')
print(data_wpm_sort)

# Plot the size of vocabulary
# Return evenly spaced values within a given interval.
# Stop at len(data_words)
# y_pos = np.arange(len(data_words))
# plt.subplot(1, 2, 1) # plt.subplot (nrows, ncols, index)
# plt.barh(y_pos, data_unique_sort.unique_words, align='center')
# plt.yticks(y_pos, data_unique_sort.comedian)
# plt.title('Number of Unique Words', fontsize=20)

# plt.subplot(1, 2, 2)
# plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
# plt.yticks(y_pos, data_wpm_sort.comedian)
# plt.title('Number of Words Per Minute', fontsize=20)

# plt.tight_layout()
# plt.show()

# Let's isolate just profanity
data_bad_words = data.transpose()[['fucking', 'fuck', 'shit']]
# Manually combine fucking and fuck as the same word
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, data_bad_words.shit], axis=1)
data_profanity.columns = ['f_word', 's_word']
print(data_profanity)

# Let's create a scatter plot of our findings
plt.rcParams['figure.figsize'] = [10, 8]  # Set width to 10 inches and height to 8 inches

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x + 1.5, y + 0.5, full_names[i], fontsize=10)  # Offset the label to avoid overlap of names and dot
    plt.xlim(-5, 155)

plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()