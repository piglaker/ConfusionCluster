
import os
import sys

from transformers import (
    AutoTokenizer, 

)

import lib


def shabb_reader(path):
    """
    """
    result = []
    try:
        with open(path) as f:
            for line in f.readlines():
                #result.append(line.replace(" ", ""))
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
    def __init__():

        return


    def _load_confusionset(self):
        path = "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/data/confusion_set/confusion.txt"




        return


class SIGHANReader(BaseReader):
    def __init__():
        return


