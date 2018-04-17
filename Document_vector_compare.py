from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.util import ngrams
import random, nltk, math, itertools
from collections import defaultdict, Counter

'''

Creates vectors of documents and compares their similarity


'''


doc_names = gutenberg.fileids()
docs = [list(gutenberg.words(doc_name)) for doc_name in gutenberg.fileids()]
vocabulary = list(itertools.chain.from_iterable(docs))
d = 100

def random_vector():
    v = [0 for _ in range(d)]
    i = 0
    while i < 5:
        index = random.randint(0,len(v)-1)
        if v[index] != 1:
            v[index] = 1
            i += 1  
    return v

index_vector = { word.lower(): random_vector() for word in vocabulary }
context_vector = { doc_name: [0.0]*d for doc_name in doc_names }

def add_vector(a, b):
    for i,x in enumerate(b):
        a[i] += x

        
for doc_name in doc_names:
    focus = doc_name
    
    for word in list(gutenberg.words(doc_name)):
        add_vector(context_vector[focus], index_vector[word.lower()])
    

def normalize(a):  
    pow2s = [math.pow(c, 2) for c in a] 
    pow2sum = sum(pow2s)
    total = math.sqrt(pow2sum)
    
    return [x/total for x in a]


def cosinedistance(a, b):
    temp = [x*y for x,y in zip(a,b)]
    cos_sim = sum(temp)
    return 1 - cos_sim


def pairwise_distance(doc_names):
    distances = []
    for doc_name1 in doc_names:
        for doc_name2 in doc_names:
            if doc_name1 != doc_name2 and doc_names.index(doc_name2) > doc_names.index(doc_name1):
                distances.append('%-8s %-8s %.3f' % (doc_name1, doc_name2, cosinedistance(
                    normalize(context_vector[doc_name1]),
                    normalize(context_vector[doc_name2]))))
    for pair in sorted(distances, key=lambda str: float(str.split()[2])):
        print(pair)

pairwise_distance(doc_names)

## Books of the same genre are "close relatives" when they are written by the same author, but otherwise they don't seem to be very close to top 15
## Almost all books written by the same author are "close relatives". Books by Chesterton seem to be most similar.
##
## Closest one to "King James Bible" is "Poems of William Blake".
## The furthest one away from "King James Bible" is "The Adventures of Buster Bear"(but it was also Alice in Wonderland in another run).
## That makes somewhat sense since Blake's poems seem to have some religious themes as well as the bible obviously.
## The narrating style and genre are very different to "The Adventures of Buster Bear" so that also makes sense.
