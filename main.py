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

def main():
    # Args

    # Model

    name_list = [ 
        ""
        "hfl/chinese-bert-wwm-ext", \
        "hfl/chinese-roberta-wwm-ext", \
        "hfl/chinese-macbert-base", \
        "hfl/chinese-xlnet-base", \
        "junnyu/ChineseBERT-base", \
        "hfl/chinese-electra-180g-base-discriminator", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/Dot_datasetsighan_ReaLiSe_eval15_epoch10_bs128", \
        #"/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_ReaLiSe/MaskedLM_datasetsighan_ReaLiSe_eval15_epoch10_bs128/checkpoint-22210", \
        #"junnyu/ChineseBERT-base" \
        ]

    name_list_2 = [
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/bert/checkpoint-5560", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/roberta/checkpoint-5560", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/macbert/checkpoint-5560", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/xlnet/checkpoint-5560", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/chinesebert/checkpoint-5560", \
        "/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/tmp/sighan_raw/ConfusionCluster/electra/checkpoint-3180", \
    ]


    name_list_3 = [
        "ReaLiSe"
    ]

    name = name_list[0]
 
    #name = name_list_2[4]

    #name = name_list_3[-1]

    print("Model:", name)

    if name in name_list_3:
        model = name # we hack
    elif 'chinesebert' in name or "ChineseBert" in name:
        config = ChineseBertConfig.from_pretrained(name)
        model = ChineseBertForMaskedLM.from_pretrained(name, config=config)
    
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
