from transformers import AutoTokenizer
from collections import defaultdict
import math
from math import log

class BPE2:

    def __init__(self, text = None, training_file = None):

        if(training_file is not None):
            file = open(training_file + '.txt', 'r', encoding='utf-8')
            training_data = file.readlines()
            self.corpus = self.to_corpus(training_data)
        else:
            self.corpus = text
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
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


    def get_tokenized_corpus(self):
        return self.tokenized_corpus
    
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

        self.count_tokens()

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


    def count_tokens(self):
        for split in self.splits.values():
            for token in split:
                self.token_counts[token] = self.token_counts.get(token, 0) + 1
        for token in self.vocab:
            if(token not in self.token_counts.keys()):
                self.token_counts[token] = 0
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

    def get_token_freqs(self):
        return self.token_counts
        

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

class UniTok2:

    def __init__(self, training_file = None):

        self.tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

        if(training_file is not None):
            file = open(training_file + '.txt', 'r', encoding = 'utf-8')
            training_data = file.readlines()
            self.corpus = self._to_corpus(training_data)

        else:
            self.corpus = [
                "this", "is", "the", "hugging", "face", "course.",
                "this", "chapter", "is", "about", "tokenization.",
                "this", "section", "shows", "several", "tokenizer", "algorithms.",
                "hopefully,", "you", "will", "be", "able", "to", "understand", "how", "they", "are", "trained", "and", "generate", "tokens.",
            ]
        
        b = BPE2(text = self.corpus)
        b.train(250)

        self.memoir = {}

        self.tokenized_corpus = b.get_tokenized_corpus()
        self.tokens = b.vocab       
        self.token_freqs = self.count_token_freqs()
        print(self.token_freqs)
        self.word_freqs = self.word_freq()
        self.tot_toks = sum(self.token_freqs.values())

        # self.char_freqs = defaultdict()
        # self.subwords_freqs = defaultdict()
        self.total_sum = b.tot_tok
        self.model = {}
        for token, freq in self.token_freqs.items():
            # if(token == 'this'): print(token, freq, self.token_freqs)
            if(freq != 0):##Bad?
                self.model[token] = -log(freq / self.total_sum)
            else:
                self.model[token] = 0
        

    def count_token_freqs(self):
        freqs = {}
        for word in self.corpus:
            for token in self.tokens:
                if(token in word):
                    freqs[token] = freqs.get(token, 0) + 1
        return freqs

    def train(self, vocabsize = 100):
        self.pare(vocabsize)

    def eval(self, input_text = None, new_file = None):
        p_val = 0
        if(new_file is not None):
            tokenized_file = self.tokenize(filename = new_file)
            p_val = self.calc(tokenized_file)
        elif(input_text is not None):
            tokenized_file = self.tokenize(text = input_text)
            p_val = self.calc(tokenized_file)
        else:
            p_val = self.calc()

        return p_val
        
    def calc(self, tokenized_file = None): 
        if(tokenized_file != None):
            scores = self.compute_scores(tokenized_file)
        else:
            scores = self.compute_scores(self.model)
        p = 0
        tot_toks = sum(freq for freq in scores.values())

        for token, freq in scores.items():
            # print(freq)
            p += log(freq/tot_toks)
        return p
        
    def _to_corpus(self, td):
        corpus = []
        for line in td:
            for word in line.split():
                corpus.append(word.lower())
        return corpus
    
    def word_freq(self):
        word_freqs = defaultdict(int)
        for word in self.corpus:
            word_freqs[word] = word_freqs.get(word, 0) + 1

        return word_freqs

    # def token_freq(self):
    #     for word, freq in self.word_freqs.items():
    #         for i in range(len(word)):
    #             self.char_freqs[word[i]] = self.char_freqs.get(word[i], 0) + freq
    #             # Loop through the subwords of length at least 2
    #             for j in range(i + 2, len(word) + 1):
    #                 self.subwords_freqs[word[i:j]] = self.subwords_freqs.get(word[i:j], 0) + freq

    #     # Sort subwords by frequency
    #     sorted_subwords = sorted(self.subwords_freqs.items(), key=lambda x: x[1], reverse=True)
    #     sorted_subwords[:10]

    # [('▁t', 7), ('is', 5), ('er', 5), ('▁a', 5), ('▁to', 4), ('to', 4), ('en', 4), ('▁T', 3), ('▁Th', 3), ('▁Thi', 3)]

        # self.token_freqs = list(self.char_freqs.items()) + sorted_subwords[: 300 - len(self.char_freqs)]
        # self.token_freqs = {token: freq for token, freq in self.token_freqs}





    # def model1(self):  

    #     # self.total_sum = sum([freq for token, freq in self.token_freqs.items()])
    #     self.model = {token: -log(freq / self.total_sum) for token, freq in self.token_freqs.items()}



    def encode_word(self, word, model):
        if(word in self.memoir.keys()):
            return self.memoir[word]
        test = False
        test2 = False
        test3 = False
        # if(word == 'this' or word == 'tokens.'): test = True
        best_segmentations = [{"start": 0, "score": 1}] + [
            {"start": None, "score": None} for _ in range(len(word))
        ]
        # print("here", word)
        for start_idx in range(len(word)):
            if(test): print("\n", start_idx)
            # This should be properly filled by the previous steps of the loop
            best_score_at_start = best_segmentations[start_idx]["score"]
            if(test): print(best_score_at_start)
            for end_idx in range(start_idx + 1, len(word) + 1):
                # if(test): print("here2")
                token = word[start_idx:end_idx]
                if(test): print(token)
                if(test): print(model['this'])
                if token in model and best_score_at_start is not None:
                    if(test2): print(token, best_score_at_start)
                    if(test2): print(model[token])
                    score = model[token] + best_score_at_start
                    # If we have found a better segmentation ending at end_idx, we update
                    if(test): print(score, best_segmentations[end_idx]["score"])
                    if (
                        best_segmentations[end_idx]["score"] is None
                        or best_segmentations[end_idx]["score"] > score
                    ):
                        if(test): print("here")
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}
                if(test): print(end_idx, best_segmentations[end_idx])
        segmentation = best_segmentations[-1]
        if(test): print(segmentation)
        if segmentation["score"] is None:
            # We did not find a tokenization of the word -> unknown
            return ["<unk>"], None

        score = segmentation["score"]
        start = segmentation["start"]
        end = len(word)
        tokens = []
        while start != 0:
            tokens.insert(0, word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        if(test3): print(tokens, score)

        self.memoir[word] = tokens, score

        return tokens, score




    # (['H', 'o', 'p', 'e', 'f', 'u', 'll', 'y'], 41.5157494601402)
    # (['This'], 6.288267030694535)



    def compute_loss(self, model):
        loss = 0
        #print("Cl")
        i = 0
        for word, freq in self.word_freqs.items():
            # print(self.token_freqs)
            # print(self.model)

                
            print(i)
            i += 1
            _, word_loss = self.encode_word(word, model)
            #if(word == 'this'): print("HERE\n", word, freq, word_loss)
            print(word)
            if(word_loss is not None):
                loss += freq * word_loss

        # print(loss)
        return loss





    def compute_scores(self, model):
        scores = {}
        model_loss = self.compute_loss(model)
        for token, score in model.items():
            # We always keep tokens of length 1
            # print(token)
            if len(token) == 1:
                continue
            # model_without_token = copy.deepcopy(model)


            score = model[token]
            del model[token]
            scores[token] = self.compute_loss(model) - model_loss
            model[token] = score
        
        # print("comp scores")
        # print(scores["ll"])
        # print(scores["his"])

        # 6.376412403623874
        # 0.0
        return scores




    def pare(self, vocabsize):
        #print(self.token_freqs)
        percent_to_remove = 1
        # print(len(self.model), vocabsize)
        k = 0
        while len(self.model) > vocabsize and k < 1:
            k += 1
            scores = self.compute_scores(self.model)
            for j in range(5000):
                print("here")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            for j in range(5000):
                print("here")
            # print(scores, "\n", sorted_scores)
            # Remove percent_to_remove tokens with the lowest scores.
            for i in range(int(len(self.model) * percent_to_remove)):
                # print(sorted_scores[i])
                # print(self.token_freqs)
                if(sorted_scores[i][0] in self.token_freqs):
                    _ = self.token_freqs.pop(sorted_scores[i][0])
                if(sorted_scores[i][0] in self.tokens):
                    _ = self.tokens.pop(self.tokens.index(sorted_scores[i][0]))
                # else:
                #     print(sorted_scores[i][0])
                #     print(self.token_freqs)

            self.token_freqs = self.count_token_freqs()


            total_sum = sum([freq for token, freq in self.token_freqs.items()])
            # self.model = {token: -log(freq / total_sum) for token, freq in self.token_freqs.items()}
 
            for token, freq in self.token_freqs.items():
                # if(token == 'this'): print(token, freq, self.token_freqs)
                if(freq != 0):##Bad?
                    self.model[token] = -log(freq / total_sum)
                else:
                    self.model[token] = 0


    def tokenize(self, model, words = None, filename = None):
        if(filename is not None):
            file = open(filename + '.txt', 'r')
            training_data = file.readlines()
            text = self._to_corpus(training_data)
        else:
            text = words.split()

        # words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        tokenized_text = []
        for word in text:
            tokenized_word, value = self.viterbi_segment(word.lower(), self.model)
            tokenized_text += tokenized_word

        return tokenized_text

        # pre_tokenized_text = self.tokenize_text(text, model)
        # encoded_words = [self.encode_word(word, model)[0] for word in pre_tokenized_text]
        # return sum(encoded_words, [])

    # def tokenize(self, text = None, filename = None):
        
    #     if(filename is not None):
    #         file = open(filename + '.txt', 'r')
    #         training_data = file.readlines()
    #         test = self._to_corpus(training_data)

    #     else:
    #         test = text.split()

    #     tokenized_test = []

    #     for word in test:
    #         tokenized_test += self.tokenize_by_tokens(word, self.tokens)

    
    # def tokenize_by_tokens(self, word, tokens):
    #     pos_tokenizations = []
    #     if(len(word) == 0 or len(word) == 1):
    #         return [word]
    #     for token in tokens:
    #         if token in word:
    #             i = word.index(token)
    #             pre = self.tokenize_by_tokens(word[0:i], self.tokens)
    #             post = self.tokenize_by_tokens(word[i+len(token)-1], self.tokens)
    #             if(pre != None and post != None):
    #                 poss = pre + [token] + post
    #                 pos_tokenizations.append(poss)
    #     best = self.viterbi_segment(word, pos_tokenizations)
    #     return best

    # def get_best_tokenization(self, word, tokenizations):
    #     self.tot_toks = sum(self.token_freqs.values())
    #     scores = [self.viterbi_segment(word, t) for t in tokenizations]
    #     best_index = scores.index(max(scores))
    #     return tokenizations[best_index]

    # def score_tokenization(self, word, tokens):
    #     sum = 0
    #     for token in tokens:
    #         if(token not in self.token_freqs.keys()):
    #             print(token)
    #         if(len(token)>0):
    #             sum += -log(self.token_freqs[token] / self.tot_toks)
    #     return sum
    
    def viterbi_segment(self, text, P):
        """Find the best segmentation of the string of characters, given the
        UnigramTextModel P."""
        # best[i] = best probability for text[0:i]
        # words[i] = best word ending at position i
        n = len(text)
        words = [''] + list(text)
        best = [1.0] + [0.0] * n
        ## Fill in the vectors best, words via dynamic programming
        for i in range(n+1):
            for j in range(0, i):
                w = text[j:i]
                if(w in P.keys()):
                    if P[w] * best[i - len(w)] >= best[i]:
                        best[i] = P[w] * best[i - len(w)]
                        words[i] = w
        ## Now recover the sequence of best words
        sequence = []; i = len(words)-1
        while i > 0:
            sequence[0:0] = [words[i]]
            i = i - len(words[i])
        ## Return sequence of best words and overall probability
        return sequence, best[-1]

        


            






        

        # words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        # pre_tokenized_text = [word for word, offset in words_with_offsets]
        # encoded_words = [self.encode_word(word, self.model)[0] for word in pre_tokenized_text]
        # return sum(encoded_words, [])
    
    def print_output(self):
        print(

        #self.tokenizer,
        # "corpus",
        # self.corpus, "\n", "\n",

        # "word_freqs",
        # self.word_freqs,"\n", "\n",

        # "char_freqs",
        # self.char_freqs,"\n", "\n"

        # "subwords_freqs",
        # self.subwords_freqs,"\n", "\n",

        "token_freqs",
        self.token_freqs,"\n", "\n",

        # "total_sum",
        # self.total_sum,"\n \n",

        # "model",
        # self.model, "\n \n"

        # "tokenized corpus",
        # self.tokenized_corpus
        )


# print(tokenize("This is the Hugging Face course.", model))

# ['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']