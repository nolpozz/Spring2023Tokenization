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

        "vocab",
        self.vocab,"\n", "\n",

        # "splits",
        # self.splits,  "\n", "\n",

        # "token counts",
        # self.token_counts,

        # "tot_tok", 
        # self.tot_tok
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


        

    def tokenize(self, text_file):

        file = open(text_file + '.txt', 'r')
        text_data = file.readlines()


        text_list = self.to_corpus(text_data)

        for text in text_list:
            pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])
    

class UniTok:

    def __init__(self, training_file):

        file = open(training_file + '.txt', 'r')
        training_data = file.readlines()



        self.corpus = self._to_corpus(training_data)
        self.b = BPE2(training_file)

        self.tokenized_corpus = {}
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # self.word_freqs = defaultdict(int)
        # self.alphabet = []
        # self.vocab = []
        # self.splits = {}

        # self.token_counts = {}
        # self.tot_tok = 0
        # self.token_uni_prob = {}

    def pre_tokenize(self, start_vocab_num):
        self.b.train(start_vocab_num)

    def calc_tok_probs(self):
        self.b.count_tokens()
    
    def _to_corpus(self, td):
        corpus = []
        for line in td:
            for word in line.split():
                corpus.append(word.lower())
        return corpus
    
    def tokenize(self):
        for word in self.corpus:
            self.tokenized_corpus[word] = self._most_prob_split(word)

    def _most_prob_split(self, word):
        pos_toks = self._get_tokenizations(word)
        probs = {}
        for possibility in pos_toks:
            probs[possibility] = self._probability(possibility)
        max_key = max(probs, key=probs.values())
        return (max_key, probs[max_key])

    def _probability(self, lst_of_toks):
        p = 1
        for tok in lst_of_toks:
            cur_p = self.b.token_counts[tok] / self.b.tot_tok
            p *= cur_p
        return p





        
    def _get_tokenizations(self, input_word):
        results = []
        stack = [(input_word, [])]
        
        while stack:
            word, current_tokens = stack.pop()
            if not word:
                results.append(current_tokens)
                continue
            for token in self.b.vocab:
                if word.startswith(token):
                    new_word = word[len(token):]
                    new_tokens = current_tokens + [token]
                    stack.append((new_word, new_tokens))
        
        return results
