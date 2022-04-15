
import torch

from dataclasses import dataclass


@dataclass
class FoolDataCollatorForSeq2Seq:
    """
    """

    def __call__(self, features):
        """
        """
        from copy import deepcopy

        f_copy = deepcopy(features)

        shared_max_length = max([ len(i['input_ids']) for i in f_copy] + [len(i['labels']) for i in f_copy] )

        def simple_pad(f_copy, key):
            f_key = [ f[key] for f in f_copy ]
            if f_key is not None:
                max_length = max(len(l) for l in f_key)

                padding_side = "right"

                if key == "attention_mask":
                    label_pad_token_id = 0
                elif key == "input_ids":
                    label_pad_token_id = 0
                elif key == "labels":
                    max_length = shared_max_length
                    label_pad_token_id= -100
                else:
                    label_pad_token_id = self.label_pad_token_id 

                for f in f_copy: 
                    remainder = [label_pad_token_id] * (max_length - len(f[key]))
                    f[key] = (
                        f[key] + remainder if padding_side == "right" else remainder + f[key]
                    )
            
            return f_copy

        for key in f_copy[0].keys():#["input_ids", "labels", "attention_mask"]:
            f_copy = simple_pad(f_copy, key)

        new = {}

        black_list = []

        for key in f_copy[0].keys():  
            new[key] = []
        
        for feature in f_copy:
            for key in feature.keys():
                new[key].append(feature[key])

        for key in new.keys():
            new[key] = torch.tensor(new[key]) 

        return new


