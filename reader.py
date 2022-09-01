
from enum import EnumMeta
import os
import sys
import re
import pickle

from numpy import isin, iterable, source

from transformers import (
    AutoTokenizer, 
    BertTokenizer, 
)

import torch
from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig


from tqdm import tqdm 

from similar_score import calcuate
import lib


def shabb_reader(path):
    """
    """
    result = []
    try:
        with open(path) as f:
            for line in f.readlines():
                line = re.sub("\n", "", line)
                result.append(line)
        return result
    except Exception as e:
        print(e)
        return 

    return 

class BaseReader():
    def __init__(self, ):
        """
        """
        # Tokenizer    
        self.tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_model_name_path 
        )
        
        self.data_collator = lib.FoolDataCollatorForSeq2Seq()#my data collator  fix the length for bert.


class ConfusionSetReader(BaseReader):
    def __init__(self, ):
        """
        """
        self.raw_data = self._load_confusionset() 
        
        self.confusion = None

        self.graph = None
        
        self.vocab = None

    def __getitem__(self, k):
        assert self.graph, "Error: Empty Graph" 
        if k in self.graph:
            return self.graph[k]
        else:
            return []

    def run(self):
        """
        """
        self.build_confusion(self.raw_data)    
    
        self.build_graph()

        self.init_vocab()

        #self.visulization()

    def build_confusion(self, data):
        """
        """
        print("[INFO] [Reader] [Building Confusion]")
        self.confusion = {}

        for key_values_string in data:
            key, values = key_values_string.split(":")

            self.confusion[key] = values
    
    def build_graph(self):
        """
        """
        print("[INFO] [Reader] [Building Graph]")
        graph = {i:[] for i in list(set([ j for i in self.confusion.values() for j in i ] + list(self.confusion.keys())))}

        for main_key in self.confusion.keys():
            line = re.sub("\W*", "", self.confusion[main_key])

            keys = [i for i in line]

            graph[main_key] += keys

            for key in keys:
                graph[key].append(main_key)

        for key in graph.keys():
            graph[key] = list(set(graph[key]))

        self.graph = graph
        
    def init_vocab(self):
        """
        """
        assert self.graph, "Error: Empty Graph"
        print("[INFO] [Reader] [Init Vocab]")
        self.vocab = {}
        for i, key in enumerate(self.graph.keys()):
            self.vocab[key] = i

    def visulization(self):
        """
        """
        print("Deprecated")
        exit()

        assert self.vocab, "Error: Empty Vocab"
        print("[INFO] [Reader] [Visulization]")

        import networkx as nx
        import matplotlib.pyplot as plt 
        G = nx.Graph()

        node_numbder = len(self.graph.values())

        H = nx.path_graph(node_numbder)
        G.add_nodes_from(H)

        all_edges = []

        for key, values in tqdm(self.graph.items()):
            for value in values:
                all_edges.append( (self.vocab[key], self.vocab[value], {'weight':1}) )
        
        G.add_edges_from( all_edges ) 

        nx.draw(G, with_labels=False, edge_color='b', node_color='g', node_size=10)
        
        plt.show()

        plt.savefig('./confusion_graph.png', dpi=300)

    def _load_confusionset(self):
        """
        """
        default_path = "./confusion.txt"

        data = shabb_reader(path=default_path)

        return data



class SighanReader(BaseReader):
    def __init__(self, tokenizer_model_name_path):
        """
        """
        #
        self.confusion_reader = ConfusionSetReader()

        self.confusion_reader.run()

        print("[INFO] [Reader] ", tokenizer_model_name_path) 

        self.tokenizer_model_name_path = tokenizer_model_name_path#"junnyu/ChineseBERT-base" 

        self.is_chinesebert = ( 'chinesebert' in self.tokenizer_model_name_path or "ChineseBert" in self.tokenizer_model_name_path )
        
        self.is_ReaLiSe = ( self.tokenizer_model_name_path.find("ReaLiSe") >= 0 )

        self.input_token = "input_ids" if not self.is_ReaLiSe else "src_idx"
        self.label_token = "labels" if not self.is_ReaLiSe else "tgt_idx" 

        print("is_chinese", self.is_chinesebert)
        print("is_ReaLiSe", self.is_ReaLiSe )

        if self.is_chinesebert:
            self.tokenizer = ChineseBertTokenizerFast.from_pretrained("junnyu/ChineseBERT-base" )#tokenizer_model_name_path)
        elif self.is_ReaLiSe:
            self.tokenizer = BertTokenizer.from_pretrained("/remote-home/xtzhang/CTC/CTC2021/milestones/ReaLise/output_holy")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        path_head = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/raw"

        path_ReaLiSe = "/remote-home/xtzhang/CTC/CTC2021/milestones/ReaLise/data"

        train_source_path = path_head + "/train.src"
        train_target_path = path_head + "/train.tgt"

        test13_source_path = path_head + "/test13.src"
        test13_target_path = path_head + "/test13.tgt"
        valid_source_path = path_head + "/valid14.src"
        valid_target_path = path_head + "/valid14.tgt"#valid should be same to test ( sighan 15
        test_source_path = path_head + "/test.src"
        test_target_path = path_head + "/test.tgt"

        train_source = []#shabb_reader(train_source_path)#[:2000]#[274144:]#for only sighan
        train_target = []#shabb_reader(train_target_path)#[:2000]#[274144:]

        #train_source = shabb_reader(train_source_path)#[:2000]#[274144:]#for only sighan
        #train_target = shabb_reader(train_target_path)#[:2000]#[274144:]

        test13_source = shabb_reader(test13_source_path)
        test13_target = shabb_reader(test13_target_path)

        valid_source = shabb_reader(valid_source_path)
        valid_target = shabb_reader(valid_target_path)

        test_source = shabb_reader(test_source_path)
        test_target = shabb_reader(test_target_path)

        all_source = train_source + test13_source + valid_source + test_source
        all_target = train_target + test13_target + valid_target + test_target

        all_data = [ (all_source[i], all_target[i]) for i in range(len(all_source))]

        print("[INFO] [Reader] [Build Map]")
        path = "map_dict2.pkl"

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.map_dict = pickle.load(f)
        else:

            self.map_dict = {}

            for src_tgt in tqdm(all_data):
                src, tgt = src_tgt
                for i in range(len(src)):
                    if src[i] != tgt[i]:
                        key = (src[i], tgt[i])

                        self.map_dict[key] = []

            with open(path, 'wb') as f:
                pickle.dump(self.map_dict, f)
        
        if self.is_ReaLiSe:
            self.test13_ReaLiSe = pickle.load(open(path_ReaLiSe+"/test.sighan13.pkl", 'rb'))
            self.test14_ReaLiSe = pickle.load(open(path_ReaLiSe+"/test.sighan14.pkl", 'rb'))
            self.test15_ReaLiSe = pickle.load(open(path_ReaLiSe+"/test.sighan15.pkl", 'rb'))

        self.source = [ i[:128] for i in all_source ]
        self.target = [ i[:128] for i in all_target ]

        self.data = all_data

        # map_dict = { (‘火','人') : [ () , (), () ] ....  }
     
        self.source_set = None
        self.target_set = None

        self.ground_truth = None

        self.init_dataset()


    def get_confusion_set(self, _source):

        confusion = [ ]

        for i, element in enumerate(_source):
            src, tgt = [ o for o in element[self.input_token]], [ o for o in element[self.label_token]]
            new = [ j for j in element[self.input_token] ]
            for j, char in enumerate(src):
                if char != tgt[j]:
                    if self.tokenizer.decode(char) in self.confusion_reader.confusion:
                        import random
                        confusion_x = random.choice(self.confusion_reader.confusion[self.tokenizer.decode(char)])
                        new[j] = self.tokenizer.convert_tokens_to_ids(confusion_x)
                    else:
                        new[j] = random.randint(671, 7662)

            confusion.append(new) 

        return confusion

    def init_dataset(self):
        """
        """
        print("[INFO] [Reader] [Init_dataset]")
        
        if self.is_chinesebert :
            path = "encodings_chinesebert.pkl"
        elif self.is_ReaLiSe:
            path = "encodings_realise.pkl"
        else:
            path = "encodings_raw.pkl"

        if os.path.exists(path):
            with open(path, 'rb') as f:
                print("Load Cache ...", path)
                self.encoding = pickle.load(f)

        elif self.is_ReaLiSe:
            
            self.encoding = {}
            _source = self.test13_ReaLiSe + self.test14_ReaLiSe + self.test15_ReaLiSe

            new_source = []
            import collections
            for i, e in enumerate(_source):
                tmp = {}
                for k,v in e.items():
                    if isinstance(v, collections.Iterable):
                        tmp[k] = v[:128]
                    else:
                        tmp[k] = 128
                new_source.append(tmp)

            _source = new_source

            self.encoding["source"] = _source
        
            _target = []
            self.encoding["target"] = self.source2target(_source)

            masked = [ ]
            for i, element in enumerate(_source):
                src, tgt = [ o for o in element[self.input_token]], [ o for o in element[self.label_token]]
                new = [ j for j in element[self.input_token] ]
                for j, char in enumerate(src):
                    if char != tgt[j]:
                        new[j] = 103
                masked.append(new) 
            from copy import deepcopy
            self.masked = deepcopy(_source)
            for i in range(len(self.masked)):
                self.masked[i][self.input_token] =  masked[i]
            self.encoding["masked"] = self.masked

            confusion = self.get_confusion_set(_source)
            self.confusion = deepcopy(_source)
            for i in range(len(self.confusion)):
                self.confusion[i][self.input_token] =  confusion[i]

            self.encoding["confusion"] = self.confusion

            with open(path, 'wb') as f:
                pickle.dump(self.encoding, f)

        else:
            self.encoding = {}
            
            self.masked = self.mask(self.source)

            if self.is_chinesebert:
                source_set = self.tokenizer(self.source, padding=True, truncation=True, max_length=128)
                target_set = self.tokenizer(self.target, padding=True, truncation=True, max_length=128)
                masked_set = self.tokenizer(self.masked, padding=True, truncation=True, max_length=128) 
            else :
                source_set = self.tokenizer.batch_encode_plus(self.source, return_token_type_ids=False)#seems transformers max_length not work
                target_set = self.tokenizer.batch_encode_plus(self.target, return_token_type_ids=False)
                masked_set = self.tokenizer.batch_encode_plus(self.masked, return_token_type_ids=False)

            source_set["labels"] = target_set[self.input_token]
            target_set["labels"] = target_set[self.input_token]
            masked_set["labels"] = target_set[self.input_token]

            truncation = not ( self.is_chinesebert )

            self.encoding["source"] = self.transpose(source_set, truncation=truncation)
            self.encoding["target"] = self.transpose(target_set, truncation=truncation)
            self.encoding["masked"] = self.transpose(masked_set, truncation=truncation)
            
            confusion = self.get_confusion_set(self.encoding["source"])
            from copy import deepcopy
            self.confusion = deepcopy(self.encoding["source"])
            for i in range(len(self.confusion)):
                self.confusion[i][self.input_token] =  confusion[i]
            
            self.encoding["confusion"] = self.confusion

            with open(path, 'wb') as f:
                pickle.dump(self.encoding, f)

        #self.masked = self.tokenizer.batch_decode([i['input_ids'] for i in self.encoding['masked']])

        #self.calculate_groundtruth()

    def get_dataset(self):
        # ["source"] ["target"] ["masked"] ["confusion"]
        return self.encoding

    def calculate_groundtruth(self):
        """
        """
        print("[INFO] [Reader] [Calculate GroundTruth]")
        calcuator =  calcuate()

        path = "ground_truth2.pkl"

        if os.path.exists(path):
            print("[INFO] [Reader] [Loading ground truth]")
            with open(path, 'rb') as f:
                self.ground_truth = pickle.load(f)
        else:

            self.ground_truth = {}

            for k, v in tqdm(self.map_dict.items()):
                score = calcuator.similar(k[0],k[1])
                self.ground_truth[k] = score

            with open(path, 'wb') as f:
                pickle.dump(self.ground_truth, f)

    def mask(self, source):
        """
        """
        masked = []
        for i in range(len(source)):
            src, tgt = [ j for j in source[i] ], [ j for j in self.target[i] ]
            new = [ j for j in source[i] ]
            for j in range(len(src)):
                if src[j] != tgt[j]:
                    new[j] = "[MASK]"
            
            masked.append("".join(new))

        return masked

    def transpose(self, inputs, truncation=True):
        features = []
        for i in tqdm(range(len(inputs[self.input_token]))):
            #ugly fix for encoder model (the same length
            max_lenth = 128 if truncation else 1000000
            features.append({key:inputs[key][i][:max_lenth] for key in inputs.keys()}) #we fix here (truncation 
        return features

    def hack(self):
        """
        hack for ReaLiSe and other model and their fucking codes
        """
        source_tok, target_tok, masked_tok = self.encoding["source"], self.encoding["target"], self.encoding["masked"]

        length = len(source_tok)

        def convert(raw, tok):
            new = []
            for i in range(length):
                tmp = {
                    'id':None, \
                    'src':raw[i], \
                    'tgt':self.target[i], \
                    'tokens_size':[ 1 for i in range(len(raw[i])) ], \
                    'src_idx':tok[i]['input_ids'], \
                    'tgt_idx':self.encoding['target'][i]['labels'], \
                    'lengths':len(raw[i])
                }

                new.append(tmp)
            return new


        new_source, new_target, new_masked = convert(self.source, source_tok), convert(self.target, target_tok), convert(self.masked, masked_tok)

        print(new_source)

        with open("./models/realise/source.pkl", "wb") as f:
            pickle.dump(new_source, f)

        with open("./models/realise/masked.pkl", "wb") as f:
            pickle.dump(new_masked, f)

    def source2target(self, source):
        target = []
        for feature in source:
            tmp = {}
            tmp["src_idx"] = feature["tgt_idx"][:128]
            tmp["tgt_idx"] = []#feature["tgt_idx"][:128]
            tmp["attention_mask"] = ([1] * len(tmp["src_idx"]))[:128]#feature["lengths"])[:128]                
            target.append(tmp)
        
        return target

def test():
    """
    """

    Reader = SighanReader("ReaLiSe")

    #Reader.init_dataset()

    #Reader.hack()

    return 


if __name__ == "__main__":
    test()

