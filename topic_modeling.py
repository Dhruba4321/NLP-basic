import pandas as pd
import pickle
from gensim import matutils, models
import scipy.sparse
import numpy as np
from nltk import word_tokenize, pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_pickle('dtm_stop.pkl')
print(data)

#make it a term-document matrix
tdm = data.transpose()
print(tdm.head())

# We're going to put the term-document matrix into a new gensim format,
# from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

# Gensim also requires a dictionary of all the terms and their respective location
# in the term-document matrix
# CountVectorizor creates dtm
cv = pickle.load(open("cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes.

# *Note: gensim refers to it as corpus, we call it term-document matrix
lda = models.LdaModel(corpus=corpus,
                      id2word=id2word,
                      num_topics=4,
                      passes=10,
                      random_state=np.random.RandomState(seed=10))

for topic, topwords in lda.show_topics():
    print("Topic", topic, "\n", topwords, "\n")

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN' # pos = part-of-speech
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)


data_clean = pd.read_pickle('data_clean.pkl')
# Apply the nouns function to the transcripts to filter only nouns
data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))
print(data_nouns)

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cv_nouns = CountVectorizer(stop_words=stop_words)
data_cv_nouns = cv_nouns.fit_transform(data_nouns.transcript)
data_dtm_nouns = pd.DataFrame(data_cv_nouns.toarray(), columns=cv_nouns.get_feature_names())
data_dtm_nouns.index = data_nouns.index
print(data_dtm_nouns)

# Create the gensim corpus - this time with nouns only
corpus_nouns = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtm_nouns.transpose()))

# Create the vocabulary dictionary with all terms and their respective location
id2word_nouns = dict((v, k) for k, v in cv_nouns.vocabulary_.items())
# Let's start with 4 topics
lda_nouns = models.LdaModel(corpus=corpus_nouns, num_topics=4, id2word=id2word_nouns, passes=10)

for topic, topwords in lda_nouns.show_topics():
    print("Topic", topic, "\n", topwords, "\n")

# Create a function to pull out nouns and adjectives from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)

# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(data_clean.transcript.apply(nouns_adj))
print(data_nouns_adj)

# Create a new document-term matrix using only nouns and adjectives,
# also remove common words with max_df;
# remove if a word appears in more than 80% of the documents.
cv_nouns_adj = CountVectorizer(stop_words=stop_words, max_df=0.8)
data_cv_nouns_adj = cv_nouns_adj.fit_transform(data_nouns_adj.transcript)
data_dtm_nouns_adj = pd.DataFrame(data_cv_nouns_adj.toarray(), columns=cv_nouns_adj.get_feature_names())
data_dtm_nouns_adj.index = data_nouns_adj.index

# Create the gensim corpus
corpus_nouns_adj = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtm_nouns_adj.transpose()))

# Create the vocabulary dictionary
id2word_nouns_adj = dict((v, k) for k, v in cv_nouns_adj.vocabulary_.items())

# Let's try 3 topics
lda_nouns_adj = models.LdaModel(corpus=corpus_nouns_adj, num_topics=3, id2word=id2word_nouns_adj, passes=10)
for topic, topwords in lda_nouns_adj.show_topics():
    print("Topic", topic, "\n", topwords, "\n")

# Keep it at 4 topics, but experiment with other hyper-parameters:
lda_nouns_adj_model = models.LdaModel(corpus=corpus_nouns_adj,
                                      num_topics=4,
                                      id2word=id2word_nouns_adj,
                                      passes=80)

for topic, topwords in lda_nouns_adj_model.show_topics():
    print("Topic", topic, "\n", topwords, "\n")

# Looking at which topics each transcript contains
corpus_transformed = lda_nouns_adj_model[corpus_nouns_adj]
print(list(zip([a for [(a, b)] in corpus_transformed], data_dtm_nouns_adj.index)))
