
import os
import sys
import re
import pickle

from numpy import source

from transformers import (
    AutoTokenizer, 

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
        print("[INFO] [Reader] ", tokenizer_model_name_path) 

        self.tokenizer_model_name_path = tokenizer_model_name_path#"junnyu/ChineseBERT-base" 

        self.is_chinesebert = ( 'chinesebert' in self.tokenizer_model_name_path or "ChineseBert" in self.tokenizer_model_name_path )
        
        print("is_chinese", self.is_chinesebert)

        if self.is_chinesebert:
            self.tokenizer = ChineseBertTokenizerFast.from_pretrained("junnyu/ChineseBERT-base" )#tokenizer_model_name_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        path_head = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/rawdata/sighan/std"

        #path_head = "./data/std"

        train_source_path = path_head + "/train.src"
        train_target_path = path_head + "/train.tgt"
        valid_source_path = path_head + "/valid14.src"
        valid_target_path = path_head + "/valid14.tgt"#valid should be same to test ( sighan 15
        test_source_path = path_head + "/test.src"
        test_target_path = path_head + "/test.tgt"

        train_source = []#shabb_reader(train_source_path)#[:2000]#[274144:]#for only sighan
        train_target = []#shabb_reader(train_target_path)#[:2000]#[274144:]

        #train_source = shabb_reader(train_source_path)#[:2000]#[274144:]#for only sighan
        #train_target = shabb_reader(train_target_path)#[:2000]#[274144:]

        valid_source = shabb_reader(valid_source_path)
        valid_target = shabb_reader(valid_target_path)

        test_source = shabb_reader(test_source_path)
        test_target = shabb_reader(test_target_path)

        all_source = train_source + valid_source + test_source
        all_target = train_target + valid_target + test_target

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
        

        self.source = [ i[:128] for i in all_source ]
        self.target = [ i[:128] for i in all_target ]

        self.data = all_data

        # map_dict = { (‘火','人') : [ () , (), () ] ....  }
     
        self.source_set = None
        self.target_set = None

        self.ground_truth = None

        self.init_dataset()

    def init_dataset(self):
        """
        """
        print("[INFO] [Reader] [Init_dataset]")
        
        if self.is_chinesebert :
            path = "encodings3.pkl"
        else:
            path = "encodings2.pkl"

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.encoding = pickle.load(f)
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

            source_set["labels"] = target_set["input_ids"]
            target_set["labels"] = target_set["input_ids"]
            masked_set["labels"] = target_set["input_ids"]

            def transpose(inputs, truncation):
                features = []
                for i in tqdm(range(len(inputs["input_ids"]))):
                    #ugly fix for encoder model (the same length
                    max_lenth = 128 if truncation else 1000000
                    features.append({key:inputs[key][i][:max_lenth] for key in inputs.keys()}) #we fix here (truncation 

                return features

            truncation = not ( self.is_chinesebert )

            self.encoding["source"] = transpose(source_set, truncation=truncation)
            self.encoding["target"] = transpose(target_set, truncation=truncation)
            self.encoding["masked"] = transpose(masked_set, truncation=truncation)

            #with open(path, 'wb') as f:
            #    pickle.dump(self.encoding, f)

        #self.masked = self.tokenizer.batch_decode([i['input_ids'] for i in self.encoding['masked']])

        #self.calculate_groundtruth()


    def get_dataset(self):
        return self.encoding["source"], self.encoding["target"], self.encoding["masked"]

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



def test():
    """
    """

    Reader = SighanReader()

    Reader.init_dataset()

    Reader.hack()

    return 


if __name__ == "__main__":
    test()

