# We'll start by reading in the corpus, which preserves word order
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import math

data = pd.read_pickle('corpus.pkl')
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

data['full_names'] = full_names
# print(data)
# print(data.columns)

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

# Apply a function along an axis of the DataFrame.
data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)
# print(data.columns)

# Create a simple scatter plot of Polarity and Subjectivity

# plt.rcParams['figure.figsize'] = [10, 8]

#for index, comedian in enumerate(data.index):
    # x = data.polarity.loc[comedian]
    # y = data.subjectivity.loc[comedian]
    # plt.scatter(x, y, color='blue')
    # plt.text(x+.001, y+.001, full_names[index], fontsize=10) # Offset the label to avoid overlap of label & dot
    # plt.xlim(-.01, .12)

# plt.title('Sentiment Analysis', fontsize=20)
# plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
# plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

# plt.show()

# Split each routine into 10 parts
def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts'''
    length = len(text)  # Calculate length of text
    size = math.floor(length / n)  # Calculate size of each chunk of text
    # Calculate the starting points of each chunk of text
    # numpy.arange([start, stop, [step])
    # Return evenly spaced values within a given interval.
    start = np.arange(0, length, size)

    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece] + size])
    return split_list

# Let's create a list of lists that'll hold all of the pieces of text of all the comedians
list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)

# The list has 12 items, one for each transcript
print(len(list_pieces))

# And then each transcript has been split into 10 pieces of text
print(len(list_pieces[0]))

# Calculate the polarity for each piece of text

polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)

print(polarity_transcript)

# Show the plot for one ali wong over time
# plt.plot(polarity_transcript[0])
# plt.title(data['full_names'].index[0])
# plt.show()

# Show the plot for all comedians
plt.rcParams['figure.figsize'] = [20, 16]

for index, comedian in enumerate(data.index):
    plt.subplot(3, 4, index + 1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0, 10), np.zeros(10))
    plt.title(data['full_names'][index])
    plt.ylim(ymin=-.2, ymax=.3)

plt.show()







