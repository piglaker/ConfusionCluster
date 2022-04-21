
import os
import sys
import re

from transformers import (
    AutoTokenizer, 

)

from tqdm import tqdm 


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

        self.visulization()

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




class SIGHANReader(BaseReader):
    def __init__(self, ):
        """
        """
        
        
        return



def test():
    """
    """

    Reader = ConfusionSetReader()

    Reader.run()

    return 


if __name__ == "__main__":
    test()

