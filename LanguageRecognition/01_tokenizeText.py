import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import *
from nltk.corpus import stopwords
from string import punctuation

text = "Mary had a little lamb. Her fleece was white as snow"


sentences = sent_tokenize(text)
print("Split into sentences.. \n")
print (sentences) # List of sentences

words = [word_tokenize(sent) for sent in sentences]
print("\n\nSplit into words.. \n")
print(words) # List of senteces -> list of words [[mary, had, a ,little, lamb, .], [her, fleece ... ]]

#list of all stop words and puncutation. (a, had, etc...)
customStopWords = set(stopwords.words('english') + list(punctuation))

wordsWithOutStopWords = [word for word in word_tokenize(text) if word not in customStopWords]
print("\n\nRemoved stop words and puncutation... \n")
print(wordsWithOutStopWords)


#identify bigrams 
bigram_measure = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWithOutStopWords)
print (sorted(finder.ngram_fd.items()))

