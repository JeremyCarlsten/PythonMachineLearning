import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.lancaster import LancasterStemmer

text2 = "Mary closed on closing night when she was in the mood to close."

stemmer = LancasterStemmer()

stemmedWords = [stemmer.stem(word) for word in word_tokenize(text2)]

print("chops the extra bits off of words, closed, closing, close -> clos")
print(stemmedWords)

print("tag words as verb, noun, etc.")
print(nltk.pos_tag(word_tokenize(text2)))

