import json
import numpy as np
import random
import dynet as dy
import heapq


class FeedForwardLanguageModel:

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=0.000001)
        self.W = self.model.add_parameters((hidden_dim, input_dim))
        self.b = self.model.add_parameters((hidden_dim,))
        self.U = self.model.add_parameters((output_dim, hidden_dim))
        self.d = self.model.add_parameters((output_dim,))

    def training(self, input, output):

        for input_, gold in zip(input, output):
            dy.renew_cg()
            input_ = dy.inputVector(input_)
            h = dy.tanh(self.W * input_ + self.b)
            pred = self.U * h + self.d
            i = list(gold).index(1)
            loss = dy.pickneglogsoftmax(pred, i)
            loss.backward()
            self.trainer.update()


    def pick_neg_log(self, pred, gold):
        if not isinstance(gold, int) and not isinstance(gold, np.int64):
            dy_gold = dy.inputVector(gold)
            return -dy.sum_elems(dy.cmult(dy_gold, dy.log(pred)))
        return -dy.log(dy.pick(pred, gold))

    def generate_poem(self,max_length=15, line_num=4):

        lines = []
        pre_word = word2index[START]
        probabilities = []
        for line in range(line_num):
            words = ""
            current_word = None
            while current_word != word2index[END]:
                x_predict = np.zeros(len(words_and_one_hot_vectors))
                x_predict[pre_word] = 1
                prob_lst = self.predict_next(x_predict)
                value = random.random()
                total = 0
                for i in range(len(word2index)):
                    total += prob_lst[i]
                    if value < total:
                        current_word = i
                        probabilities.append(prob_lst[i])
                        break
                if current_word == word2index[END]:
                    break
                word = index2word[current_word]
                if len(words.split()) >= max_length:
                    break
                words += word + " "
                pre_word = current_word
            lines.append(words)

        poem = ""
        for line in lines:
            line = line.split()
            for word in line:
                if word == 'NEWLINE':
                    poem += '\n'
                else:
                    poem += word + " "
        print(poem)
        self.perplexity(probabilities)
        return poem


    def generate_poem_cumulative(self, max_length, line_num):

        lines = ""
        pre_word = word2index[START]
        probabilities = []
        newline = 0
        words = ""
        current_word = None
        line_lenght = 0
        while newline < line_num:
            x_predict = np.zeros(len(words_and_one_hot_vectors))
            x_predict[pre_word] = 1
            prob_lst = self.predict_next(x_predict)
            value = random.random()
            total = 0
            for i in range(len(word2index)):
                total += prob_lst[i]
                if value < total:
                    current_word = i
                    probabilities.append(prob_lst[i])
                    break

            if current_word == word2index['NEWLINE']:
                words += " \n"
                newline = newline + 1
            else:
                word = index2word[current_word]
                word_with_blank = word + " "
                words += word_with_blank
            pre_word = current_word


            line_lenght = len(words.split())
            if line_lenght > max_length:
                lines += words + "\n"
                newline = newline + 1
                words = ""
                pre_word = word2index["NEWLINE"]



            if current_word == word2index[END]:
                break


        print(lines)

        self.perplexity(probabilities)
        return words

    def generate_poem_softmax(self,linenum):

        poem = []
        pre_word = word2index[START]
        probabilities = []
        newline = 0
        words = ""
        line_max_word = 0

        while newline <= linenum:

            x_predict = np.zeros(len(words_and_one_hot_vectors))
            x_predict[pre_word] = 1
            max_probabilities,softmax = self.predict(x_predict)

            if pre_word == word2index[START] or pre_word == word2index['NEWLINE']:
                current_word =  np.random.choice(max_probabilities)
            else:
                current_word = max(max_probabilities)

            if pre_word == current_word:
                current_word = np.random.choice(max_probabilities)

            if pre_word == word2index['NEWLINE']:
                words += '\n'
                newline = newline + 1
            else:
                word = index2word[current_word]
                word_with_blank = word + " "
                words += word_with_blank
            prob = softmax[current_word]
            probabilities.append(prob)
            pre_word = current_word


            if current_word == word2index[END]:
                break

            line_max_word = line_max_word + 1

            if line_max_word > 10:
                line_max_word = 0
                pre_word = word2index['NEWLINE']
                newline = newline + 1
                words += '\n'

        poem.append(words)
        print(words)
        self.perplexity(probabilities)
        return poem


    def predict_next(self, x):
        x = dy.inputVector(x)
        h = dy.tanh(self.W * x + self.b)
        predicted = self.U * h + self.d
        softmax = np.exp(predicted.npvalue()) / np.sum(np.exp(predicted.npvalue()))
        return softmax

    def predict(self,x):
        x = dy.inputVector(x)
        pred = ((self.U * dy.tanh(self.W * x + self.b))) + self.d
        softmax = dy.softmax(pred).npvalue()
        max_pos = heapq.nlargest(20, range(len(softmax)), key=softmax.__getitem__)
        return max_pos, softmax

    def perplexity(self,probabilities):
        p = 0
        for x in probabilities:
            p += np.log2(x)
        p = -p
        p = p /len(probabilities)
        p = np.power(2,p)
        print(p)



START = '<s>'
END = '</s>'
word2index= {}
bigrams = []
unique = []

def make_bigram(words):
    words = words.split()
    for x in range(len(words)-1):
        if words[x] == END:
            return
        bigrams.append([words[x],words[x+1]])
        unique.append(words[x])
        unique.append(words[x+1])

    return



def read():

    with open('unim_poem.json', 'r') as myfile:
        data = myfile.read()

    poems = json.loads(data)

    count_poem = 0

    for my_poem in poems:

        if count_poem < 10:
            lines = my_poem["poem"]
            lines = START + " " + lines + " " + END
            lines = lines.splitlines()

            p = []
            for words in lines:
                words= words + " NEWLINE"
                p.append(words)
            poem = " ".join(p)
            make_bigram(poem)


        count_poem = count_poem + 1



read()
unique = set(unique)
unique = list(unique)


i = 0
for unigram in unique:
    if not unigram in word2index:
        word2index[unigram] = i
        i = i + 1



words_and_one_hot_vectors = {}


def generate_one_hot_vector(i, word):
    a = np.zeros(len(word2index), 'uint8')
    a[i] = 1
    words_and_one_hot_vectors[word] = a
    return a


words = list(word2index.keys())


for x in range(len(word2index)):
    generate_one_hot_vector(x, words[x])



input = []
output = []


for bigram in bigrams:
    input.append(words_and_one_hot_vectors[bigram[0]])
    output.append(words_and_one_hot_vectors[bigram[1]])

index2word = {}

for word in word2index:
    index2word[word2index[word]] = word

word_num = len(words_and_one_hot_vectors)


poem_model = FeedForwardLanguageModel(input_dim=word_num, hidden_dim=10, output_dim=word_num)

print("model training...")
for i in range(50):
    poem_model.training(input, output)
    print(i+1,"st. epoch is finished")

poem_model.model.save('model1024_5000poem_epoch50.d2v')



print("SOFTMAX")
poem_1 = poem_model.generate_poem_softmax(5)
poem_2 = poem_model.generate_poem_softmax(5)
poem_3 = poem_model.generate_poem_softmax(5)
poem_4 = poem_model.generate_poem_softmax(5)
poem_5 = poem_model.generate_poem_softmax(5)

# print("CUMUTLATIVE")
cpoem_1 = poem_model.generate_poem_cumulative(10,5)
cpoem_2 = poem_model.generate_poem_cumulative(10,5)
cpoem_3 = poem_model.generate_poem_cumulative(10,5)
cpoem_4 = poem_model.generate_poem_cumulative(10,5)
cpoem_5 = poem_model.generate_poem_cumulative(10,5)


