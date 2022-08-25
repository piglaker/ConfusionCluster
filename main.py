import os
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List

import numpy as np
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import (
    AutoModel,
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers import trainer_utils, training_args
from transformers.trainer_pt_utils import nested_detach
from transformers import BertForMaskedLM, BartForConditionalGeneration
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig

import reader
import model
import runner
from lib import FoolDataCollatorForSeq2Seq

import sys


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    # Args
    topk = 5

    # Model

    name_list = [ 
        ""
        "hfl/chinese-bert-wwm-ext", \
        "hfl/chinese-roberta-wwm-ext", \
        #"hfl/chinese-macbert-base", \
        #"hfl/chinese-xlnet-base", \
        "junnyu/ChineseBERT-base", \
        #"hfl/chinese-electra-180g-base-discriminator", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/Dot_datasetsighan_ReaLiSe_eval15_epoch10_bs128", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/MaskedLM_datasetsighan_ReaLiSe_eval15_epoch10_bs128/checkpoint-22210", \
        #"junnyu/ChineseBERT-base" \
        ]

    name_list_2 = [
 
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/3/bert", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/3/roberta", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/macbert", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/xlnet", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/3/chinesebert", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/electra", \

         "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/2/bert", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/2/roberta", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/macbert", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/xlnet", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/2/chinesebert", \

        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/bert", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/roberta", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/macbert", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/xlnet", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/chinesebert", \

    ]


    name_list_3 = [
        "ReaLiSe",
        "ReaLiSe_holy",
        "PLOME_holy",
    ]

    name_list_4 = [
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_holy/ConfusionCluster/2/bert", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_holy/ConfusionCluster/2/roberta", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_holy/ConfusionCluster/2/nezha", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_holy/ConfusionCluster/2/chinesebert", \

        
    ]

    name_list_5 = [
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_mask/ConfusionCluster/3/bert", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_mask/ConfusionCluster/3/roberta", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_mask/ConfusionCluster/3/chinesebert", \
        
    ]

    name_list_6 = [
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_holy_mask/ConfusionCluster/3/bert",

    ]


    name_list_6 = [
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/Proto/macbert/Proto_cls_copy0_cl0.007_repeat1_eval15_epoch20_bs48_seed3471_multi_taskFalse_v1",   
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/Proto/macbert/Proto_cls_copy0_cl0.01_repeat1_eval15_epoch20_bs48_seed3471_multi_taskFalse_weight0.01_v1",
    ]

    #name = name_list[0]
 
    #name = name_list_2[2]

    #name = name_list_3[-1]

    #name = name_list_4[2]

    #name = name_list_5[2]

    name = name_list_6[-1]

    output_path = "./logs/"+ name.replace("/", "_") + "_topk_" + str(topk) +".txt"

    print("output_path:", output_path)

    sys.stdout = Logger(output_path)

    print("Model:", name)

    if name in name_list_3:
        model = name # we hack
    elif 'chinesebert' in name or "ChineseBert" in name:
        config = ChineseBertConfig.from_pretrained(name)
        model = ChineseBertForMaskedLM.from_pretrained(name, config=config)
    elif "Proto" in name:
        from models.modeling_bert_v4 import ProtoModel_v3 as ProtoModel
        import torch
        model = ProtoModel("hfl/chinese-macbert-base", None)
        model.load_state_dict(torch.load("/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/Proto/macbert/Proto_cls_copy0_cl0.007_repeat1_eval15_epoch20_bs48_seed3471_multi_taskFalse_weight0.05_v1/pytorch_model.bin"))
    else:
        model = BertForMaskedLM.from_pretrained(name)

    #model = BartForConditionalGeneration.from_pretrained(name)

    #model = AutoModel.from_pretrained(name)

    sighan_reader = reader.SighanReader(name)

    confusion_reader = reader.ConfusionSetReader()

    # Data Collator
    data_collator = FoolDataCollatorForSeq2Seq()#my data collator  fix the length for bert.

    # Runner
    Runner = runner.ConfusionClusterRunner(
        model=model,
        args=None,#training_args,         
        reader=sighan_reader,
        confusion_reader=confusion_reader,
        data_collator=data_collator,
        topk=topk,    
    )

    # Run
    run_result = Runner.run()

    # Train
    #train_result = runner.train()

    # Evaluate
    #evaluate_result = runner.evaluate()

    # Predict
    #predict_result = runner.predict()



    return

if __name__ == "__main__":
    main()
