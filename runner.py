


from collections.abc import Mapping
from numpy import source

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


from tqdm import tqdm



class BaseRunner():
    def __init__():
        
        return

class ConfusionClusterRunner():
    def __init__(self, model, args, reader, data_collator):
        """
        """
        self.args = args
        
        self.dataset = None
        
        self.cluster_algorithm = None
        
        self.model = model
        
        self.confusionset = None
        
        self.reader = reader
        
        self.data_collator = data_collator

        self.device = 'cuda'

        self.batch_size = 64

        return

    def Hungary_Algorithm(self):
        """
        """
        return

    def single_step(self, batch):
        """
        """

        BertOutput = self.model(
            **batch,
            return_dict=True, 
            output_hidden_states=True,  
        )

        return BertOutput.hidden_states[-1].detach().cpu()

    def forward(self):
        """
        """
        print("[INFO] [Runner] [Forward]")


        source_set, target_set = self.reader.get_dataset()

        source_host = []

        self.model.eval()

        self.model.to(self.device)

        source_dataloader = DataLoader(
                source_set,
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
        )

        target_dataloader = DataLoader(
                target_set,
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
        )

        for inputs in tqdm(source_dataloader ):
            inputs = self._prepare_inputs(inputs)
            hiddens = self.single_step(inputs)

            source_host += hiddens

        target_host = []

        for inputs in tqdm( target_dataloader ):
            inputs = self._prepare_inputs(inputs)
            hiddens = self.single_step(inputs)

            target_host += hiddens

        return source_host, target_host

    def calculate_similarity(self, hiddens):
        """
        calculate Model's Prediction
        """
        print("[INFO] [Runner] [Calculate_Similarity] ")

        source_hiddens, target_hiddens = hiddens

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        for i in tqdm(range(len(self.reader.data))):
            source, target = self.reader.source[i], self.reader.target[i]

            for j in range(len(source)):
                if source[j] != target[j]:
                    a = source_hiddens[i][j]
                    b = target_hiddens[i][j]
                    self.reader.map_dict[(source[j], target[j])].append(cos(a, b))

        return

    def calcuate_score(self):
        """
        calculate the result comparing Model's Prediction with Ground Truth
        """


        return


    def run(self):
        """
        """

        # forward 
        hiddens = self.forward()

        # similarity
        cluster_result = self.calculate_similarity(hiddens)

        for k,v in self.reader.map_dict.items():
            print(k, v)
            #break

        # Score
        score = self.calculate_score(cluster_result)
 
        return

    def train(self):
        """
        """
        return


    def _prepare_input(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        return self._prepare_input(inputs)


