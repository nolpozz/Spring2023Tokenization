from transformers import AutoTokenizer
from collections import defaultdict
import math

class WordTok:

    
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    def __init__(self, training_file):

        file = open(training_file + '.txt', 'r', encoding='utf-8')
        training_data = file.readlines()

        self.corpus = self.to_corpus(training_data)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.word_freqs = defaultdict(int)
        self.alphabet = []
        self.vocab = []
        self.splits = {}

        self.start()

        self.token_counts = {}
        self.tot_tok = 0
        self.token_uni_prob = {}


    
    def start(self):
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] = self.word_freqs.get(word, 0) + 1

        for word in self.word_freqs.keys():
            if word[0] not in self.alphabet:
                self.alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in self.alphabet:
                    self.alphabet.append(f"##{letter}")

        self.alphabet.sort()

        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + self.alphabet.copy()

        self.splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freqs.keys()
        }

    def to_corpus(self, td):
        corpus = []
        for line in td:
            for word in line.split():
                corpus.append(word.lower())
        return corpus
    
    def compute_pair_scores(self, splits):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores
    
    # def vis_pair_scores(self):
    #     pair_scores = self.compute_pair_scores(self.splits)
    #     for i, key in enumerate(pair_scores.keys()):
    #         print(f"{key}: {pair_scores[key]}")
    #         if i >= 5:
    #             break

    def best_pair_score(self):
        best_pair = ""
        max_score = None
        pair_scores = self.compute_pair_scores(self.splits)
        for pair, score in pair_scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score

        return (best_pair, max_score)
    

    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
    

    # def run(self):

    #     best_pair, max_score = self.best_pair_score()
    #     self.vocab.append(best_pair)
    #     self.splits = self.merge_pair(best_pair[0], best_pair[1], self.splits)

    def train(self, vocab_size):
        while len(self.vocab) < vocab_size:
            scores = self.compute_pair_scores(self.splits)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            self.splits = self.merge_pair(*best_pair, self.splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)
    
    def print_vocab(self):
        print(self.vocab)

    def print_output(self):
        print(

        #self.tokenizer,
        "corpus",
        self.corpus, "\n", "\n",

        "word_freqs",
        self.word_freqs,"\n", "\n",

        "alphabet",
        self.alphabet,"\n", "\n",

        # "merges",
        # self.merges,"\n", "\n",

        "vocab",
        self.vocab,"\n", "\n",

        "splits",
        self.splits)


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
        # values = list(self.merges.values())
        # for i in range(len(values)):
        #     merge = values[-i]
        #     print(merge)
        #     if merge in word:
        #         sub = self.tokenized_word(word)
        #         tokened_word = [merge] + sub
        #         print(merge, sub)
        # return tokened_word
        pos = len(word)
        while(pos>=1):
            if(word[0:pos] == "\""):
                sub = self.tokenized_word(word[pos:])
                return ["\""] + sub
            if(word[0:pos] in self.vocab):
                sub = self.tokenized_word(word[pos:])
                if(sub is not None):
                    return [word[0:pos]] + sub
            pos -= 1
        return None




    def tokenize(self, text_file):

        file = open(text_file + '.txt', 'r', encoding = 'utf-8')
        text_data = file.readlines()


        test_corpus = self.to_corpus(text_data)


        tokenized_text = []
        for word in test_corpus:
            tokenized_word = self.tokenized_word(word)
            if(tokenized_word is not None):
                tokenized_text += tokenized_word
        return tokenized_text
