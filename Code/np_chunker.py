# this code is for creating and tagging noun phrases
from string import punctuation
import stanza
import numpy as np

def noun_phraser(sentence, np_range=1):

    # here we initialize n as the number of words in the sentence
    n = len(sentence.words)

    # we also initialize a hash mapping to keep track of words that are already on a noun phrase
    hasher = np.zeros(n)

    # tagging the words as we go along 0 means O and 1 means I
    # however we cannot use ADPs that are shorter than 3 characters
    useless_adps = []
    for w in sentence.words:
        if (w.upos == 'ADP' and len(w.text) <= 3) or w.upos == 'AUX':
            useless_adps.append(w.id - 1)

    obi_tags = np.zeros(n)

    # this list will keep the noun phrases caught within the sentence
    noun_phrases = []

    # now we iterate through the words
    for i in range(n):

        # if the current word was not processed before
        if hasher[i] == 0:

            # the ith word
            word = sentence.words[i]

            # counting the words in the noun phrase
            word_counter = 0

            # this well keep the current noun phrase
            current_np = ''

            # may be the first rule didn't apply so we check the hash again
            if hasher[i] == 0:

                word = sentence.words[i]

                # this is a quadruple rule
                if word.id < n and (word.upos == 'NOUN' or word.upos == 'ADJ')\
                        and sentence.words[word.id].head == word.head and \
                        word.head == word.id + 3 and (sentence.words[word.head - 1].upos != 'PUNCT') and \
                        (sentence.words[word.id].upos == 'NOUN'):  # or sentence.words[word.id].upos == 'PROPN'):

                    current_np += word.text + " " + sentence.words[word.id].text + ' ' \
                                  + sentence.words[word.head - 2].text + sentence.words[word.head - 1].text
                    # add to the current noun phrase

                    obi_tags[word.id] = 1
                    obi_tags[word.head - 1] = 1
                    noun_phrases.append(current_np)
                    hasher[i] = 1
                    hasher[i + 1] = 1
                    hasher[i + 2] = 1
                    hasher[i + 3] = 1

                # this is a triplet rule
                # deleted word.head == word.id + 2 and
                elif word.id < n and (word.upos == 'NOUN' or word.upos == 'ADJ')\
                        and sentence.words[word.id].head == word.head and \
                        word.head == word.id + 2 and (sentence.words[word.head - 1].upos != 'PUNCT') and \
                        (sentence.words[word.id].upos == 'NOUN'):  # or sentence.words[word.id].upos == 'PROPN'):

                    current_np += word.text + " " + sentence.words[word.id].text + ' '\
                                  + sentence.words[word.head - 1].text  # add to the current noun phrase
                    obi_tags[word.id] = 1
                    obi_tags[word.head - 1] = 1
                    noun_phrases.append(current_np)
                    hasher[i] = 1
                    hasher[i+1] = 1
                    hasher[i+2] = 1

                elif word.id < n and hasher[i] == 0 and (word.upos == 'ADJ' or word.upos == 'NOUN') and sentence.words[word.id].upos == 'NOUN' \
                        and word.head == word.id + 1:  # detecting consecutive "adj + noun" or  "noun + noun" noun phrases

                    current_np += word.text + " " + sentence.words[word.id].text  # add to the current noun phrase
                    obi_tags[word.id] = 1
                    noun_phrases.append(current_np)
                    hasher[i] = 1

            # running a traversal on the noun phrase starting here
            while((hasher[word.id - 1] == 0) and (word.head <= word.id + np_range) and
                  (word.upos == 'NOUN'  or word.upos == 'ADJ' )): # or word.upos == 'PROPN' or word.upos == 'ADP') ):

                # increasing the count of the words within the NP
                word_counter += 1

                # update obi tags
                if word_counter > 1:
                    obi_tags[word.id-1] = 1

                # appending the word to the current NP
                current_np += word.text

                # here we have a special case where the word has adposition, we concatenate all
                temp_word = word
                while temp_word.id < n and  sentence.words[temp_word.id].upos == 'ADP' and len(sentence.words[temp_word.id].text) < 3:
                    temp_word = sentence.words[temp_word.id]
                    current_np += temp_word.text

                current_np += ' '

                # also hashing this position to 1
                hasher[word.id - 1] = 1

                # update the current word
                word = sentence.words[word.head - 1]

            # append the current NP to the list
            if word_counter >= 1:
                current_np = current_np.strip(punctuation)
                flagg = True
                for xx in [',', '.', ';', ']', '[', '!']:
                    if xx in current_np:
                        flagg = False
                if flagg:
                    noun_phrases.append(current_np)

    obi_tags_updated = []
    for i in range(len(obi_tags)):
        if i not in useless_adps:
            obi_tags_updated.append(obi_tags[i])

    obi_tags_updated = np.array(obi_tags_updated)

    return (noun_phrases, obi_tags)

def noun_phraser_v5(sentence):

    sent_noun_phrases = []

    for word in sentence.words:

        if word.id+1 < len(sentence.words):

            if word.upos in ['NOUN'] and sentence.words[word.head-1].head == 0:
                sent_noun_phrases.append(word.text)

            if word.upos in ['NOUN', 'ADJ', 'VERB'] and sentence.words[word.id].upos in\
               ['NOUN'] and word.id + 1 == word.head:
                sent_noun_phrases.append(word.text + ' ' + sentence.words[word.id].text)

            if word.upos in ['NOUN', 'ADJ'] and sentence.words[word.id].upos in ['NOUN', 'ADJ']\
                    and sentence.words[word.id+1].upos in ['NOUN'] and word.id + 2 == word.head\
                    and sentence.words[word.head-1].head == 0:
                sent_noun_phrases.append(word.text + ' ' + sentence.words[word.id].text + ' '
                                         + sentence.words[word.id].text)

    return sent_noun_phrases, None






