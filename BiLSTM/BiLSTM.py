import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import unicodedata
import re
import random
import time
import math
from evaluation import score
import os
import Constants
from Lang import Label, Lang
from Constants import fourway, binary, conn
import sys
from dataset import PipeData, JsonData, generate_standard_data
from masked_crossentropy import masked_cross_entropy
from Constants import parsed_explicit_data_path, PDTB_data_path


SOS = 1
EOS = 0
PADDING = 0
inf = 9999999999
USE_CUDA = torch.cuda.is_available()
print("USE_CUDA:", USE_CUDA)



def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_file(addr):
    data = []
    label = []
    maxlength = 0
    with open(addr,'r') as reader:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        for each in reader:
            line = each.strip().split(" ||| ")
            sen_1 = normalize_string(line[0]).split(' ')
            sen_2 = normalize_string(line[1]).split(' ')
            data.append([sen_1, sen_2, normalize_string(line[2])])
            lens = max(len(sen_1), len(sen_2))
            if(maxlength < lens):
                maxlength = lens
    maxlength+=1
    return data, maxlength

def index_pairs(data, lang, label, maxlength, batch_size):
    batch_1 = []
    size1_list = []
    batch_2 = []
    size2_list = []
    label_list = []
    for i in range(batch_size):
        data_one = random.choice(data)
        input_sen, input_size = index_sentence(data_one[0], lang, maxlength)
        output_sen, output_size = index_sentence(data_one[1], lang, maxlength)
        la = label.label2index[data_one[2]]
        label_list.append((la))
        batch_1.append(input_sen)
        size1_list.append(input_size)
        batch_2.append(output_sen)
        size2_list.append(output_size)

    return batch_1, size1_list, batch_2, size2_list, label_list

def index_sentence(sentence,lang, maxlength):
    # print (sentence)
    index_list = [lang.word2index[word] for word in sentence]
    index_list.append(EOS)
    index_size = len(index_list)
    temp = index_size
    while(temp < maxlength):
        index_list.append(PADDING)
        temp+=1

    return index_list, index_size

def index_sentence_evaluate(sentence,lang):
    # print (sentence)
    index_list = [lang.word2index[word] for word in sentence]
    index_size = len(index_list)
    index_list.append(EOS)
    index_tensor = torch.LongTensor(index_list)
    result = Variable(index_tensor.view(1, -1))
    return result, [index_size]

class Attn(nn.Module):
    def __init__(self, batch_size):
        super(Attn, self).__init__()
        self.batch_size = batch_size
    def forward(self,unpacked_1,unpacked_2, model ="train"):
        if model == "train":
            arg1 = torch.sum(unpacked_1, dim= 1).unsqueeze(2)
            arg2 = torch.sum(unpacked_2, dim= 1).unsqueeze(2)
            mask_1 = unpacked_1.eq(0)
            mask_2 = unpacked_2.eq(0)
            unpacked_1.masked_fill_(mask_1, -inf)
            unpacked_2.masked_fill_(mask_2, -inf)

            p_1 = F.softmax(torch.bmm(unpacked_1, arg2).view(self.batch_size, 1, -1),dim=2)
            p_2 = F.softmax(torch.bmm(unpacked_2, arg1).view(self.batch_size, 1, -1),dim=2)
        else:
            arg1 = torch.sum(unpacked_1, dim= 1).unsqueeze(2)
            arg2 = torch.sum(unpacked_2, dim= 1).unsqueeze(2)
            p_1 = F.softmax(torch.bmm(unpacked_1, arg2).view(1, 1, -1),dim=2)
            p_2 = F.softmax(torch.bmm(unpacked_2, arg1).view(1, 1, -1),dim=2)

        return p_1,p_2

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_size, out_size, n_layers= 1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.LSTM1 = nn.LSTM(input_size, hidden_size, n_layers,batch_first= True, bidirectional=True)
        self.LSTM2 = nn.LSTM(input_size, hidden_size, n_layers,batch_first= True, bidirectional=True)
        self.attn = Attn(batch_size)
        self.Linear = nn.Linear(4*hidden_size , out_size)
        if USE_CUDA:
            self.LSTM1 = self.LSTM1.cuda()
            self.LSTM2 = self.LSTM2.cuda()
            self.Linear = self.Linear.cuda()


    def forward(self,input_var1, input_var2, size1_list, size2_list, id_sort_1, id_sort_2,id_unsort_1, id_unsort_2, hidden_1, hidden_2, model= "train"):
        input_1 = self.embedding(input_var1)
        input_2 = self.embedding(input_var2)
        if USE_CUDA:
            input_1 = input_1.cuda()
            input_2 = input_2.cuda()
            id_unsort_1 = id_unsort_1.cuda()
            id_unsort_2 = id_unsort_2.cuda()
        input_1 = nn.utils.rnn.pack_padded_sequence(input_1,
                                                    list(torch.LongTensor(size1_list)[id_sort_1]),
                                                    batch_first=True)
        input_2 = nn.utils.rnn.pack_padded_sequence(input_2,
                                                    list(torch.LongTensor(size2_list)[id_sort_2]),
                                                    batch_first=True)

        out_1, hidden_1 = self.LSTM1(input_1, hidden_1)
        out_2, hidden_2 = self.LSTM2(input_2, hidden_2)
        unpacked_1 = nn.utils.rnn.pad_packed_sequence(out_1, batch_first=True)[0][id_unsort_1]
        unpacked_2 = nn.utils.rnn.pad_packed_sequence(out_2, batch_first=True)[0][id_unsort_2]
        p_1, p_2 = self.attn(unpacked_1, unpacked_2, model)
        #p_1, p_2 = self.attn(hidden_1, hidden_2, model)
        arg1_new = torch.bmm(p_1, unpacked_1).squeeze(1)
        arg2_new = torch.bmm(p_2, unpacked_2).squeeze(1)
        pair = torch.cat((arg1_new, arg2_new), dim=1 )
        output = self.Linear(pair)
        output = F.softmax(output,dim=1)
        #hidden_1[0].cpu().data.numpy()
        return  output
    def init_hidden(self):
        if USE_CUDA:
            return (Variable(torch.zeros(2*self.n_layers, self.batch_size, self.hidden_size)).cuda(),
                    Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size)),
                    Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_size))
                    )
    def init_hidden_evaluate(self):
        if USE_CUDA:
            return (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_size)).cuda(),
                    Variable(torch.zeros(2 * self.n_layers, 1, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(2 * self.n_layers, 1, self.hidden_size)),
                    Variable(torch.zeros(2 * self.n_layers, 1, self.hidden_size))
                    )

def train_batch(input_var1, input_var2, size1_list, size2_list, id_sort_1, id_sort_2, id_unsort_1,
                id_unsort_2, label_list, net, net_optimizer, conn_similarity_matrix):
    net_hidden1 = net.init_hidden()
    net_hidden2 = net.init_hidden()
    output = net(input_var1, input_var2, size1_list, size2_list, id_sort_1, id_sort_2,id_unsort_1, id_unsort_2, net_hidden1, net_hidden2)
    label_var = Variable(torch.LongTensor(label_list)).cuda() if USE_CUDA else Variable(torch.LongTensor(label_list))
    #loss = loss_func(output, label_var)
    loss = masked_cross_entropy(output, label_var, conn_similarity_matrix, label_type)
    net_optimizer.zero_grad()
    loss.backward()
    net_optimizer.step()
    return loss

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    if percent == 0:
        percent = percent + 0.0000000001
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def evaluate(sen_1, sen_2, max_length, lang, net):
    input_var1,size1_list  = index_sentence_evaluate(sen_1, lang )
    input_var2, size2_list = index_sentence_evaluate(sen_2, lang)
    _, id_sort_1 = torch.sort(torch.LongTensor(size1_list), dim=0, descending=True)
    _, id_sort_2 = torch.sort(torch.LongTensor(size2_list), dim=0, descending=True)
    _, id_unsort_1 = torch.sort(id_sort_1, dim=0)
    _, id_unsort_2 = torch.sort(id_sort_2, dim=0)
    net_hidden1 = net.init_hidden_evaluate()
    net_hidden2 = net.init_hidden_evaluate()
    output = net(input_var1, input_var2, size1_list, size2_list, id_sort_1, id_sort_2,id_unsort_1, id_unsort_2, net_hidden1, net_hidden2, model ="evaluate")
    # Run through encoder
    topkv, topki = torch.topk(output.data, 1, 1)
    ni = topki.tolist()[0][0]
    return ni


def evaluate_all(addr, maxlength, lang, label, net):
    with open(addr,'w') as writer:
        for data_one in data:
            output_label= evaluate(data_one[0], data_one[1], maxlength, lang, net)
            writer.write("label "+data_one[2]+' '+"predict "+label.index2label[output_label]+'\n')
def evaluate_all2(addr, data, maxlength, lang, label, net):
    total = 0
    good = 0.0
    predict = []
    ground = []
    f = open(addr, 'w')
    for data_one in data:
        total+=1
        output_label= evaluate(data_one[0], data_one[1], maxlength, lang, net)
        predict.append(output_label)
        if label_type == binary:
            ground.append(label_2_pridict if data_one[2].lower() == label_2_pridict else "non-"+label_2_pridict)
        else:
            ground.append(data_one[2])
        if data_one[2] == label.index2label[output_label]:
            good+=1
        f.write("label " + data_one[2] + ' ' + "predict " + label.index2label[output_label] + '\n')
    f.write(str(good / total))
    return ground,predict

def get_command_line_parameters():
    parameters = argv_list[1:]
    dict = {}
    for one_para in parameters:
        key, value = one_para.split("=")
        if key in dict:
            print("error, parameters conflict")
            print(exit())
        else:
            dict[key.strip()] = value.strip()
    return dict

if __name__ =="__main__":
    #para
    argv_length = len(sys.argv)
    argv_list = sys.argv
    para_dict = get_command_line_parameters()
    hidden_size = int(para_dict["hidden_size"])
    batch_size = int(para_dict["batch_size"])
    #n_class = int(para_dict["n_class"])
    LR = float(para_dict["LR"])
    EPOCH = int(para_dict["EPOCH"])
    print_every = int(para_dict["print_every"])
    print_every_sub = int(para_dict["print_every_sub"])
    label_type = para_dict["label_type"]
    label_2_pridict = para_dict["label_2_pridict"]

    '''
    hidden_size = 50
    batch_size = 8
    n_class = 5
    LR = 0.001
    EPOCH = 20
    print_loss_total = 0
    print_every = 5
    print_every_sub = 1000
    '''
    print_loss_total = 0
    start = time.time()
    # prepare data
    print("preparing data")
    #data, maxlength = read_file(r"data/imTrain.Sdata")
    #data_test, maxlength_test = read_file(r"data/imTest.Sdata")
    print("reading explict data")
    #read explicit data from PDTB parser
    parsed_explicit_data = PipeData(type_list=["Explicit"], path=parsed_explicit_data_path)
    parsed_explicit_data.read_pipe_data({})

    #read explicit data from PDTB
    PDTB_explicit_data = PipeData(type_list=["Explicit"], path=PDTB_data_path)
    PDTB_explicit_data.read_pipe_data({})

    #read implicit data from PDTB, 02-20 is training
    PDTB_implicit_data_training = PipeData(type_list=["Implicit"], path=PDTB_data_path)
    PDTB_implicit_data_training.read_pipe_data(["02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19"])

    # read implicit data from PDTB, 21-22 is testing
    PDTB_implicit_data_testing = PipeData(type_list=["Implicit"], path=PDTB_data_path)
    PDTB_implicit_data_testing.read_pipe_data(["21", "22"])

    #chenck whether all dataset are read correctly
    if (len(parsed_explicit_data.instances) == 0):
        print("WARNNING!! parsed_explicit_data length is zero, please check")
    if (len(PDTB_explicit_data.instances) == 0):
        print("WARNNING!! PDTB_explicit_data length is zero, please check")
    if (len(PDTB_implicit_data_training.instances) == 0):
        print("WARNNING!! PDTB_explicit_data length is zero, please check")
    if (len(PDTB_implicit_data_testing.instances) == 0):
        print("WARNNING!! PDTB_explicit_data length is zero, please check")

    #data_explicit_training, maxlength_ex = generate_standard_data([parsed_explicit_data, PDTB_explicit_data], label_type)
    data_implicit_training, maxlength_im_training = generate_standard_data([PDTB_implicit_data_training], label_type)
    data_implicit_testing, maxlength_im_testing = generate_standard_data([PDTB_implicit_data_testing], label_type)
    data = data_implicit_training#data_explicit_training + data_implicit_training
    maxlength = maxlength_im_training#maxlength_ex if maxlength_ex > maxlength_im_training else maxlength_im_training
    data_test = data_implicit_testing
    maxlength_test = maxlength_im_testing
    #data = data[0:1000]
    #data_test = data_test[0:1000]
    print("training data size:",len(data))
    print("testing data size:", len(data_test))
    lang = Lang('corpus')
    label = Label("label")

    for num, one_data in enumerate(data):
        lang.index_words(one_data[0])
        lang.index_words(one_data[1])
        label.index_labels(one_data[2], label_type=label_type, label_2_pridict=label_2_pridict)
        if label_type.lower() == "conn":
            label.record_conn_sense(one_data[2], one_data[3])#one_data[2] is always label, if label_type != conn, one_data[2] is sense, one_data[3] is conn
        else:
            label.record_conn_sense(one_data[3], one_data[4])#if label_type != conn, one_data[3] is conn, one_data[4] is sense
    for num, one_data in enumerate(data_test):
        lang.index_words(one_data[0])
        lang.index_words(one_data[1])
        label.index_labels(one_data[2], label_type=label_type, label_2_pridict=label_2_pridict)
    label.generate_similarity_matrix()
    n_class = len(label.index2label)
    print (label.index2label)
    #init net
    print("init net")
    net = BiLSTM(lang.n_words, hidden_size, hidden_size, batch_size, n_class)
    emb_file = "data/embed.pth"
    reload_glove = True
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
        if emb.size()[0] != lang.n_words:
            reload_glove = True
        else:
            reload_glove = False
    if reload_glove == True:
        # load glove embeddings and vocab
        glove_file = open('data/glove/glove.6B.50d.txt', "r")
        glove_lines = glove_file.readlines()
        glove_dict = {}
        glove_dim = 0
        for line in glove_lines:
            one_line = line.rstrip("\n").split(" ")
            word = one_line[0]
            vector = one_line[1:]
            glove_dim = len(vector)
            glove_dict[word] = torch.Tensor(list(map(float, vector)))

        emb = torch.Tensor(lang.n_words, glove_dim).normal_(-0.05, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in lang.word2index.keys():
            if word in glove_dict.keys():
                emb[lang.word2index[word]] = glove_dict[word]
        torch.save(emb, emb_file)
    net.embedding.weight.data.copy_(emb)
    net_optimizer = optim.Adam(net.parameters(),LR)
    #loss_func = nn.CrossEntropyLoss()
    #loss_func = masked_cross_entropy()

    #train
    result_matrix = open("result_matrix.txt", "w")
    for epoch in range(EPOCH):
        print(epoch)
        print_loss_total = 0
        for i in range(int((len(data))/batch_size)):
            batch_1, size1_list, batch_2, size2_list, label_list = index_pairs(data, lang, label, maxlength, batch_size)
            _, id_sort_1 = torch.sort(torch.LongTensor(size1_list), dim=0, descending=True)
            _, id_sort_2 = torch.sort(torch.LongTensor(size2_list), dim=0, descending=True)
            _, id_unsort_1 = torch.sort(id_sort_1, dim=0)
            _, id_unsort_2 = torch.sort(id_sort_2, dim=0)
            input_var1 = Variable(torch.LongTensor(batch_1)[id_sort_1]).view(batch_size, maxlength)
            input_var2 = Variable(torch.LongTensor(batch_2)[id_sort_2]).view(batch_size, maxlength)
            '''
            loss = train_batch(input_var1, input_var2, size1_list, size2_list,
                               id_sort_1, id_sort_2, id_unsort_1, id_unsort_2,
                               label_list, net, net_optimizer, loss_func)
            '''
            loss = train_batch(input_var1, input_var2, size1_list, size2_list,
                               id_sort_1, id_sort_2, id_unsort_1, id_unsort_2,
                               label_list, net, net_optimizer, label.conn_similarity_matrix)

            print_loss_total += loss
            del loss
            if i == 0: continue

            if i % print_every_sub == 0:
                print_loss_avg = print_loss_total / print_every_sub / batch_size
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (
                    time_since(start, i / (len(data)/batch_size)), i, i / (len(data)/batch_size) * 100, print_loss_avg)
                print(print_summary)

        #if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every / batch_size
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start, epoch / EPOCH), epoch, epoch / EPOCH * 100, print_loss_avg)
            print(print_summary)
        del print_loss_total
        torch.save(net, "net_attn.pkl")
        #evaluate_all('result_label.txt', maxlength, lang, label, net)
        ground, predict = evaluate_all2('result_label.txt', data_test, maxlength_test, lang, label, net)
        '''
        f = open('result_label.txt', "r")
        lines = f.readlines()
        label_list = []
        prediction_list = []
        for line in lines:
            label_list.append(line.split(" ")[1])
            prediction_list.append(line.split(" ")[3].rstrip())
        '''
        predict = [label.index2label[i] for i in predict]
        rep, res = score(ground, predict)
        #write the result of this epoch to result_matrix.txt
        result_matrix.writelines(rep)
        result_matrix.writelines("\n")
        result_matrix.writelines(str(res))
        #result_matrix.close()
        #result_matrix.writelines("\n")
        print("evaluation result:")
        print(rep)
        print(res)
        del rep, res, predict, ground


