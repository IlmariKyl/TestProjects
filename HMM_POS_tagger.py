import nltk, random, itertools, re, copy
from nltk.tag import hmm
from nltk.corpus import treebank
from collections import Counter


def read_tagged_sents(file_path):
    tagged_sents = []
    try:
        gs_file = open(file_path, "r", encoding='utf8')
        line = gs_file.readline()
        while line != "":
            tagged_sents.append([(wt.split("/")[0].lower(), wt.split("/")[1]) for wt in line.split()])
            line = gs_file.readline()
        gs_file.close()
    except OSError:
        print("Error reading file.")
  
    return tagged_sents


def main():

    file_path = r'fi-ud-train.pos-tagged.txt'

    tagged_sents = read_tagged_sents(file_path)
    random.shuffle(tagged_sents)

    # Copy the 5 first sentences so that the words that will be replaced with '<UNK>' can
    # be used when printing
    ref_sents = copy.deepcopy(tagged_sents[:5])
    
    size = int(len(tagged_sents) * 0.1)
    train_set, test_set = tagged_sents[size:], tagged_sents[:size]

    # Make 2 list variables that consist only of (w,t) tuples so the word amount is easier to count
    train_set_words = list(itertools.chain.from_iterable(train_set))
    test_set_words = list(itertools.chain.from_iterable(test_set))

    # Frequencies of words in the train_set
    train_set_wt_freqs = Counter(train_set_words)

    # Go through train_set and change words with frequencies below 3 to '<UNK>'        
    for i, sent in enumerate(train_set):
        for j, (word, tag) in enumerate(sent):
            if train_set_wt_freqs[(word,tag)] < 3:
                sent[j] = ('<UNK>', tag)
                
        if i > 500: # For the 500 first sentences
            break


    unk_words = []
    
    # Go through test_set and change words that don't appear in the train_set into '<UNK>'
    for sent in test_set:
        for i, (word, tag) in enumerate(sent):
            if (word, tag) not in train_set_words:
                unk_words.append((word, tag))
                sent[i] = ('<UNK>', tag)
                
    UNK_rel_freq = len(unk_words) / len(test_set_words)
    print("Relative frequency of unknown words in the test set: {}\n".format(UNK_rel_freq))

    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_set)

    print("HMM based POS tagger's accuracy: {}".format(tagger.evaluate(test_set)))

    # List of the 5 first sentences in the test_set
    print_sents = [[word for word, tag in tagged_sent] for tagged_sent in tagged_sents[:5]]
   
    print('\n\n5 sentences tagged by the ConsecutivePosTagger:\n')
    for i, sent in enumerate(print_sents):
        print("Sentence {}:".format(i+1))
        tagged_sent = tagger.tag(sent)

        # Add the actual word in front of the possible '<UNK>'
        for j, (word, tag) in enumerate(tagged_sent):
            ref_word = ref_sents[i][j][0]
            tagged_sent[j] = (re.sub(r'(<UNK>)', ref_word+r'\1', word), tag)

        print(tagged_sent, "\n")


main()

## The accuracy varies from 85-87% so it is pretty good. The best results I got when I
## changed the words with frequencies below 3 to <UNK>s in the train set, and did it only
## for 500 of the first sentences. When more sentences were processed, the results got
## slightly worse. 
## All of the tagged sentences seemed to be mostly correct, a few unknown words were tagged
## incorrectly in every run.



