class BPE:

    def __init__(self, training_file):

        file = open(training_file + '.txt', 'r')
        # for line in file:
        #     print(line)
        # file.close()
        training_data = file.readlines()
        #print(content[0])
        
        # self.corpus = self.start_word_freq(training_data)
        #print(self.corpus)
        self.tokens = set()
        self.splits = self.start_splits(training_data)
        self.merges = set()

    def train(self, n):
        i = 0
        while(i < n):
            self.run()
            i += 1


    def run(self):

        merge = self.most_frequent_pair(self.splits)
        self.splits = self.merge(self.splits, merge)

    
    def merge(self, splits, merge):
        splits2 = []
        for line in splits:
            print(line)
            line2 = []
            pass_here = False
            add_last = True
            for i in range(len(line) - 1):
                print(line[i], line[i+1])
                if(pass_here == True):
                    pass_here = False
                    if(i+1 == len(line)-1):
                        add_last = False
                    pass
                elif(line[i] + line[i + 1] == merge):
                        line2.append(merge)
                        pass_here = True
                else:
                    line2.append(line[i])
            if(line != []):
                if(add_last is True):
                    print(line)
                    line2.append(line[-1])
            splits2.append(line2)
        return splits2

    def most_frequent_pair(self, splits):
        freq = {}
        for line in splits:
            for i in range(len(line)-1):
                if(line[i + 1] == " "):
                    pass
                else:
                    pair = (line[i], line[i+1])
                    freq[pair] = freq.get(pair, 0) + 1
        most_freq = (max(freq.items(), key=lambda x:x[1]))[0]#key of most frequent element tuple
        pair = most_freq[0] + most_freq[1]
        self.merges.add((most_freq, pair))
        self.tokens.add(pair)
        return pair


    def start_splits(self, training_data):
        '''
        instanitalizes the splits by making a 2dlist of every character one each
        line in the data
        
        also adds each character to tokens(set)
        '''
        letter_lines = []
        for line in training_data:
            words = line.split()
            this_line = []
            for word in words:
                for letter in word:
                    this_line.append(letter)
                    self.tokens.add(letter)
                this_line.append(" ")
            letter_lines.append(this_line[0: -1])
        return letter_lines
