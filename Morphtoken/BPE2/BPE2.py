from transformers import AutoTokenizer
from collections import defaultdict
import math

class BPE2:

    def __init__(self, training_file):


        file = open(training_file + '.txt', 'r', encoding='utf-8')
        training_data = file.readlines()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.corpus = self.to_corpus(training_data)
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.get_splits()
        self.alphabet = self.get_alpha()
        self.merges = {}
        self.token_counts = {}

        self.vocab = []

        self.token_uni_prob = {}

        self.tot_tok = 0

        self.tokenized_corpus = []



    def to_corpus(self, td):
        corpus = []
        for line in td:
            for word in line.split():
                corpus.append(word.lower())
        return corpus

    def get_splits(self):
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1

    def get_alpha(self):
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        return alphabet
        
    def compute_pair_freqs(self):
        pair_freqs = {}
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        return pair_freqs

    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    def train(self, vocab_size=50):

        self.alphabet.sort()
        self.vocab = ["<|endoftext|>"] + self.alphabet.copy()
        self.splits = {word: [c.lower() for c in word] for word in self.word_freqs.keys()}

        while len(self.vocab) < vocab_size:
            pair_freqs = self.compute_pair_freqs()
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self.splits = self.merge_pair(*best_pair, self.splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
        for split in self.splits.values():
            self.tokenized_corpus += split

        

        
    
    
    def print_vocab(self):
        print(sorted(self.vocab))

    def print_merges(self):
        print(self.merges)

    def print_output(self):
        print(
        #self.tokenizer,
        # "corpus",
        # self.corpus, "\n", "\n",

        # "word_freqs",
        # self.word_freqs,"\n", "\n",

        # "alphabet",
        # self.alphabet,"\n", "\n",

        # "merges",
        # self.merges,"\n", "\n",

        # "vocab",
        # self.vocab,"\n", "\n",

        # "splits",
        # self.splits,  "\n", "\n",

        # "token counts",
        # self.token_counts,

        # "tot_tok", 
        # self.tot_tok,

        "tokenized corpus",
        self.tokenized_corpus
        )

    # def count_tokens(self):
    #     self.token_counts = {}
    #     for token in self.vocab:
    #         for split in self.splits.values():
    #             if token in split:
    #                 self.token_counts[token] = self.token_counts.get(token, 0) + 1
    #     self.tot_tok = sum(self.token_counts.values())


    def count_tokens(self):
        for split in self.splits.values():
            for token in split:
                self.token_counts[token] = self.token_counts.get(token, 0) + 1
        self.tot_tok = sum(self.token_counts.values())


    def calc_prob(self, token):
        if(token in self.token_counts.keys()):
            return self.token_counts[token] / self.tot_tok
        return None
    
    # def calc_prob(self, token):
    #     return self.token_counts.get(token, None) / self.tot_tok
        

    def compute_unigram_prob(self):
        self.count_tokens()
        for token in self.vocab:
            prob = self.calc_prob(token)
            self.token_uni_prob[token] = prob
        p_value = sum(math.log(prob) for prob in self.token_uni_prob.values() if prob is not None)
        print(p_value)

    def calc_test_prob(self, token_count, sum):
        return token_count/sum

    def count_test_tokens(self, tokenized_text):
        token_counts = {}
        for token in tokenized_text:
            token_counts[token] = token_counts.get(token, 0) + 1
        return (token_counts, sum(token_counts.values()))

    def eval(self, tokenized_text):
        test_token_prob = {}
        tokens_counts, sum = self.count_test_tokens(tokenized_text)
        for token, count in tokens_counts.items():
            prob = self.calc_test_prob(count, sum)
            test_token_prob[token] = prob
        p_value = 0
        for prob in list(test_token_prob.values()):
            p_value += math.log(prob)
        # p_value = sum(math.log(prob) for prob in list(test_token_prob.values()))
        print(p_value)
        

    # def compute_unigram_prob(self):

    #     for token in self.vocab:
    #         prob = self.calc_prob(token)
    #         self.token_uni_prob[token] = prob

    #     p_value = 0
    #     for prob in self.token_uni_prob.values():
    #         if(prob is not None):
    #             p_value += math.log(prob)

            #estimate count based probability
            #for each token    freq/total num tokens

        # print(p_value)


    #new corpus comes in 
    #test is if it represents the new corpus well(new ratio matches old ratio)


    #make sure test corpus doesn't have any characters not covered by test corpus

    #smooth data by taking ln(P) by extension ln(P1) + ln(P2) ...


    #Then do unigram tokenization


    def tokenized_word(self, word):
        print(word)
        # tokened_word = []
        if(word in self.vocab or len(word) == 1):
            return [word]
        pos = 0
        while(pos>=len(word)):
            if(word[0:pos] == "\""):
                sub = self.tokenized_word(word[pos:])
                return ["\""] + sub
            if(word[0:pos] in self.vocab):
                sub = self.tokenized_word(word[pos:])
                if(sub is not None):
                    return [word[0:pos]] + sub
            pos += 1
        return None




    def tokenize(self, text_file):

        file = open(text_file + '.txt', 'r', encoding = 'utf-8')
        text_data = file.readlines()


        test_corpus = self.to_corpus(text_data)

        # for text in text_list:
        #     pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        # pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        # splits = [[l for l in word] for word in pre_tokenized_text]

        tokenized_text = []
        for word in test_corpus:
            tokenized_word = self.tokenized_word(word)
            # print(word)
            # print(tokenized_word)
            # print(self.tokenized_word("\"project"))
            if(tokenized_word is not None):
                tokenized_text += tokenized_word
        return tokenized_text

        # for pair, merge in self.merges.items():
        #     for idx, split in enumerate(splits):
        #         i = 0
        #         while i < len(split) - 1:
        #             if split[i] == pair[0] and split[i + 1] == pair[1]:
        #                 split = split[:i] + [merge] + split[i + 2 :]
        #             else:
        #                 i += 1
        #         splits[idx] = split
        # return sum(splits, [])