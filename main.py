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
from transformers import BertForMaskedLM
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

import reader
import model
import runner
from lib import FoolDataCollatorForSeq2Seq

def main():
    # Args

    sighan_reader = reader.SighanReader()

    # Model
    model = BertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # Data Collator
    data_collator = FoolDataCollatorForSeq2Seq()#my data collator  fix the length for bert.

    # Runner
    Runner = runner.ConfusionClusterRunner(
        model=model,
        args=None,#training_args,         
        reader=sighan_reader,
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
