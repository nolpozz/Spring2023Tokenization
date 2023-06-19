import collections
import re

class Morfessor:
    def __init__(self, training_file, num_subwords):
        file = open(training_file + '.txt', 'r')
        training_data = file.readlines()
        self.corpus = self.to_corpus(training_data)
        self.num_subwords = num_subwords
        self.subwords = collections.defaultdict(int)
        self.total_words = 0
        self.word_counts = collections.defaultdict(int)
        self.subword_counts = collections.defaultdict(int)
        self.build_vocabulary()

    def to_corpus(self, td):
        corpus = []
        for line in td:
            for word in line.split():
                corpus.append(word.lower())
        return corpus

    def build_vocabulary(self):
        """Initialize the vocabulary with character n-grams"""
        for word in self.corpus:
            self.total_words += 1
            self.word_counts[word] += 1
            for i in range(len(word)):
                for j in range(i + 1, min(i + self.num_subwords + 1, len(word) + 1)):#From second letter to either the end of the word or ??
                    subword = word[i:j]#seems like an arbitrary substring
                    self.subwords[subword] += 1


    def learn_subwords(self):
            """Learn subwords from the corpus"""
            for _ in range(self.num_subwords):#number of subwords
                for word, count in self.word_counts.items():
                    subwords = self.split_word(word)
                    for subword in subwords:
                        self.subword_counts[subword] += count

                self.subwords = collections.defaultdict(int)##meant to reset?
                for subword, count in self.subword_counts.items():
                    if count >= 5:#why 5
                        self.subwords[subword] = count

                self.subword_counts = collections.defaultdict(int)#again resetting?

    def split_word(self, word):
        """Split a word into subwords"""
        if word in self.subwords:
            return [word]

        subwords = []
        for i in range(len(word)):
            for j in range(i + 1, min(i + self.num_subwords + 1, len(word) + 1)):
                subword = word[i:j]
                if subword in self.subwords:
                    subwords.extend(self.split_word(word[j:]))
                    subwords.insert(0, subword)
                    return subwords

        return [word]


    def encode_word(self, word):
            """Encode a word into subwords"""
            subwords = []
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    subword = word[i:j]
                    if subword in self.subwords:
                        subwords.append(subword)
                        break
                else:
                    subwords.append(subword)

            return subwords

    def decode_word(self, subwords):
            """Decode subwords into a word"""
            return "".join(subwords)
    

    def print_output(self):
        print(

        #self.tokenizer,
        # "corpus",
        # self.corpus, "\n", "\n",

        # "num_subwords",
        # self.num_subwords,"\n", "\n",

        "subwords",
        self.subwords,"\n", "\n",

        # "total_words",
        # self.total_words,"\n", "\n",

        # "word_counts",
        # self.word_counts,"\n", "\n",

        # "subword_counts",
        # self.subword_counts
        )