

from Constants import fourway, binary, conn
from scipy import spatial

class Lang(object):
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.index2word = {1:"SOS",0:"EOS"}
        self.word2count = {}
        self.n_words = 2

    def index_words(self,sentence):
        for word in sentence:
            if word in self.word2index:
                self.word2count[word]+=1

            else:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words += 1


class Label(object):
    def __init__(self,name):
        self.name = name
        self.label2index = {}
        self.index2label = {}
        self.n_labels = 0
        self.label2count = {}
        self.conn_sense_dict = {}
        self.senses = []
        self.conn_similarity_matrix = []

    def index_labels(self,label, label_type, label_2_pridict):
        if label_type == binary:
            if label not in self.label2index:
                self.label2index[label] = 1 if label.lower() == label_2_pridict.lower() else 0
                self.index2label[1] = label_2_pridict
                self.index2label[0] = "non-" + label_2_pridict
            converted_label = self.index2label[self.label2index[label]]
            if converted_label not in self.label2count:
                self.label2count[converted_label] = 1
            else:
                self.label2count[converted_label] += 1
        elif label_type == fourway:
            if label not in self.label2index:
                self.label2index[label] = self.n_labels
                self.index2label[self.n_labels] = label
                self.n_labels += 1
            if label not in self.label2count:
                self.label2count[label] = 1
            else:
                self.label2count[label] += 1
        elif label_type == conn:
            if label not in self.label2index:
                self.label2index[label] = self.n_labels
                self.index2label[self.n_labels] = label
                self.n_labels += 1
            if label not in self.label2count:
                self.label2count[label] = 1
            else:
                self.label2count[label] += 1

    def record_conn_sense(self, conn, sense):
        if sense not in self.senses:
            self.senses.append(sense)
        if conn not in self.conn_sense_dict:
            self.conn_sense_dict[conn] = {sense: 1}
        elif sense not in self.conn_sense_dict[conn]:
            self.conn_sense_dict[conn][sense] = 1
        else:
            self.conn_sense_dict[conn][sense] += 1

    def generate_similarity_matrix(self):
        conn_sense_matrix = self.generate_conn_vector()
        self.conn_similarity_matrix = [[0 for i in range(len(conn_sense_matrix.keys()))] for j in range(len(conn_sense_matrix.keys()))]
        keys = [key for key in conn_sense_matrix.keys()]
        for i in range(len(conn_sense_matrix.keys())):
            one_conn1 = keys[i]
            for j in range(i, len(conn_sense_matrix.keys())):
                one_conn2 = keys[j]
                vector1 = conn_sense_matrix[one_conn1]
                vector2 = conn_sense_matrix[one_conn2]
                result = 1 - spatial.distance.cosine(vector1, vector2)
                if result > 0:
                    pass
                else:
                    result = 0
                self.conn_similarity_matrix[i][j] = result
                self.conn_similarity_matrix[j][i] = result
        print("debug")


    def generate_conn_vector(self):
        conn_sense_matrix = {}
        for i in range(len(self.index2label)):
            one_conn = self.index2label[i]
            if one_conn in self.conn_sense_dict:
                one_conn_vector = [self.conn_sense_dict[one_conn][one_sense] if one_sense in self.conn_sense_dict[one_conn]
                                   else 0 for one_sense in self.senses]
            else:
                one_conn_vector = [0 for one_sense in self.senses]
            conn_sense_matrix[one_conn] = one_conn_vector
        return conn_sense_matrix
