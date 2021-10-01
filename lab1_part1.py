#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()


# In[4]:


nltk.download('gutenberg')


# In[1]:


from nltk.corpus import gutenberg
gutenberg.fileids()[0]


# In[7]:


bible = gutenberg.words('bible-kjv.txt')
bible


# In[9]:


type(bible)


# In[17]:


bible_words = list(map(str.lower,bible))


# In[19]:


words_set = set(bible_words)
word_count = [(word, bible_words.count(word)) for word in words_set]
word_count[:10]


# In[28]:


#sorting the term-occurrence relation
word_rank = ss.rankdata([w_count for (word,w_count) in word_count])
word_ranked = [(w_count[0], w_count[1], word) for w_count, word in zip(word_count, word_rank)]
print(word_ranked[0:10])
count_sorted = sorted(word_ranked, key=lambda x:x[2])
count_sorted[-10:]


# In[32]:


def symbol_removal(word):
    #removes all additional symbols and special characters from the string
    #to return a plain text word
    return re.sub('[^A-Za-z0-9\s]+', '', word).lower()


# In[36]:


stopwords = set(STOPWORDS)
freq_list = {}
for word, count in word_count:
    if word not in stopwords: 
        word = symbol_removal(word)
        if word:
            word = porter.stem(word)
            freq_list[word] = freq_list.get(word, 0) + count

word_cloud = WordCloud(background_color='white')
word_cloud.generate_from_frequencies(freq_list)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[26]:


#zipf's law
L = len(count_sorted)
X = np.array([np.log(L-rank+1) for (word, count, rank) in count_sorted])
Y = np.array([np.log(count) for (word, count, rank) in count_sorted])
plt.plot(X, Y, 'b.')
A = np.vstack([X, np.ones(L)]).T
slope, cc = np.linalg.lstsq(A, Y, rcond=None)[0]
plt.plot(X, slope*X + cc, 'r')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.show()
print(f'slope of the line: {round(slope, 2)} and corpus constant of line: {round(cc, 2)}')


# In[ ]:




