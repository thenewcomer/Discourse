

import os
import json
import unicodedata
import re


class PipeData():
    def __init__(self, type_list, path):
        self.path = path
        self.type = type_list
        self.instances = []

    def getPipeList(self, pipRoot, types, file_list):  # return a dict with instance list in one file is a domain, the key is the absolute path of this file
        ## types is a list, [Implicit, Explicit, NoRel...]
        dictPipInstances = {}
        for root1, dirs1, files1 in os.walk(pipRoot):
            if len(files1) == 0 and len(dirs1) > 0:
                if len(file_list) == 0:
                    file_list = dirs1
                for dirs_one in dirs1:
                    if dirs_one in file_list:
                        for root, dirs, files in os.walk(root1+dirs_one):
                            files = [f for f in files if f.find(".pipe") != -1]
                            for f in files:
                                fullPath = os.path.join(root, f)
                                with open(fullPath, encoding="latin1") as ff:
                                    pipContent = ff.read().strip().split("\n")
                                    pipInstances = [x.split("|") for x in pipContent]  # 2D list
                                    # filtering out unwanted type
                                    filteredInstances = [x for x in pipInstances if x[0] in types]
                                    filteredInstances = [x + [i, False] for i, x in enumerate(
                                        filteredInstances)]  # the last entry: visited, the last but two entry: index
                                # print "flag"
                                if filteredInstances != []:
                                    dictPipInstances[f] = filteredInstances
        return dictPipInstances

    def read_pipe_data(self, file_list):
        pipList = self.getPipeList(self.path, self.type, file_list)
        for pipIns_key in pipList.keys():
            for pipIns in pipList[pipIns_key]:
                dict = {}
                dict["Explicit_or_Implicit"] = pipIns[0]
                dict["Sense"] = pipIns[11]
                if dict["Sense"].lower().strip() == "":
                    continue
                if dict["Explicit_or_Implicit"] == "Explicit":
                    dict["Conn"] = pipIns[8]
                else:
                    dict["Conn"] = pipIns[9]
                dict["Arg1"] = pipIns[24]
                dict["Arg2"] = pipIns[34]
                self.instances.append(dict)


class JsonData():
    def __init__(self):
        self.instances = []

    def read_json_data(self, path):
        f = open(path)
        self.instances = json.load(f)

    def add_json_data(self, path):
        f = open(path)
        instances = json.load(f)
        self.instances = self.instances + instances



def generate_standard_data(data_class_list, label_key):
    data = []
    max_length = 0
    for data_class in data_class_list:
        for ins in data_class.instances:
            sen_1 = normalize_string(ins["Arg1"]).split(' ')
            sen_2 = normalize_string(ins["Arg2"]).split(' ')
            sense = normalize_string(ins["Sense"])
            if label_key.lower() == "conn":
                data.append([sen_1, sen_2, normalize_string(ins["Conn"]), sense])
            else:
                data.append([sen_1, sen_2, sense.split(".")[0], normalize_string(ins["Conn"]), sense])
            lens = max(len(sen_1), len(sen_2))
            if(max_length < lens):
                max_length = lens
    max_length+=1
    return data, max_length


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    return s

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )