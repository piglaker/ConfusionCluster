
import pickle

from collections.abc import Mapping
from numpy import average, source

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


from tqdm import tqdm

from similar_score import calcuate



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

        self.all_result = None

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

    def single_infer(self, batch):
        """
        """

        BertOutput = self.model(
            **batch,
            return_dict=True, 
            output_hidden_states=True,  
        )

        return torch.softmax(BertOutput.logits, 2).detach().cpu()

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

    def predict(self):
        """
        """

        print("[INFO] [Runner] [Predict]")

        source_set, _, masked_set = self.reader.get_dataset()

        source_host = []

        self.model.eval()

        self.model.to(self.device)

        source_dataloader = DataLoader(
                source_set,
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
        )

        masked_dataloader = DataLoader(
                masked_set,
                batch_size=self.batch_size,
                collate_fn=self.data_collator,
        )

        for inputs in tqdm( source_dataloader ):
            inputs = self._prepare_inputs(inputs)
            scores = self.single_infer(inputs)

            source_host += scores

        masked_host = []

        for inputs in tqdm( masked_dataloader ):
            inputs = self._prepare_inputs(inputs)
            scores = self.single_infer(inputs)

            masked_host += scores

        return source_host, masked_host

    def estimate(self, scores):
        """
        """
        print("[INFO] [Runner] [Estimate]")

        source_scores, masked_scores = scores

        all_result = []

        for i in tqdm(range(len(self.reader.data))):
            source, target = self.reader.encoding["source"][i]["input_ids"], self.reader.encoding["target"][i]["input_ids"]
            
            score = source_scores[i]
            masked_score = masked_scores[i]

            result_host = []

            for j in range(len(source)):
                x = source[j]
                y = target[j]

                if x != y:

                    noise = None

                    all_possible_noise = masked_score.topk(3, dim=1)[-1][j]
                    
                    for possible_noise in all_possible_noise:
                        if possible_noise != x and possible_noise != y:                   
                           noise = possible_noise
                           break

                    p_y_cx = score[j][y]

                    p_noise_cx = score[j][noise]

                    p_y_cm = masked_score[j][y]

                    p_noise_cm = masked_score[j][noise]

                    result = {
                        "x":x,
                        "X":self.reader.tokenizer.decode(x),
                        "y":y,
                        "Y":self.reader.tokenizer.decode(y),
                        "noise":noise,
                        "Noise":self.reader.tokenizer.decode(noise),
                        "p_y_cx":p_y_cx,
                        "p_noise_cx":p_noise_cx,
                        "p_y_cm":p_y_cm,
                        "p_noise_cm":p_noise_cm,
                    }

                    result_host.append(result)

            all_result.append(result_host)

        self.all_result = all_result

        print(all_result[:10])

        return all_result

    def analysis(self):
        """
        """
        print("[INFO] [Runner] [Analysis]")
        count = 0
        mlm_count = 0
        correct_count = 0
        for example in self.all_result:
            for result in example:
                bad_mlm = result["p_noise_cm"] > result["p_y_cm"]
                good_correction = result["p_y_cx"] > result["p_noise_cx"]
                if  bad_mlm and good_correction :
                    count += 1 
                if bad_mlm:
                    mlm_count += 1
                if good_correction:
                    correct_count += 1

        length =  sum( [ len(i) for i in self.all_result] )

        print(
            "[Bad_MLM and Good_Correction]:", count / length, \
            "[Bad_MLM]:", mlm_count / length, \
            "[Good_Correction]:", correct_count / length, \
        )

    def evaluate(self, scores):
        """
        """
        print("[INFO] [Runner] [Evaluate]")

        source_scores, masked_scores = scores

        #s = torch.tensor(source_scores)

        #print(s.shape)

        preds = [ torch.argmax(s, dim=-1) for s in source_scores[1062:] ]

        masked_preds =  [ torch.argmax(s, dim=-1) for s in masked_scores[1062:] ]

        sources = [ i["input_ids"] for i in self.reader.encoding["source"] ][1062:]

        labels = [ i["input_ids"] for i in self.reader.encoding["target"] ][1062:]

        print(sources[0])
        print(preds[0])
        print(labels[0])

        print(self.reader.tokenizer.decode(sources[0]))
        print(self.reader.tokenizer.decode(preds[0]))
        print(self.reader.tokenizer.decode(labels[0]))

        self.metric((sources, preds, labels))

        self.metric((sources, masked_preds, labels))

        return

    def metric(self, input):
        """
        """
        from core import _get_metrics
        computer = _get_metrics()
        #print(input[0])
        evaluate_result = computer(input)

        print(evaluate_result)

        return 

    def calculate_similarity(self, hiddens):
        """
        calculate Model's Prediction
        """
        print("[INFO] [Runner] [Calculate_Similarity] ")

        source_hiddens, target_hiddens = hiddens

        cos = nn.CosineSimilarity(dim=0, eps=1e-16)

        for i in tqdm(range(len(self.reader.data))):
            source, target = self.reader.encoding["source"][i]["input_ids"], self.reader.encoding["target"][i]["input_ids"]

            for j in range(len(source)):
                if source[j] != target[j]:
                    try:
                        a = source_hiddens[i][j]
                        b = target_hiddens[i][j]
                    except:
                        print(source)
                        exit()
                    key =  (self.reader.tokenizer.decode(source[j]), self.reader.tokenizer.decode(target[j]))

                    value = cos(a, b)
                    #print(value, torch.isnan(value)) 
                    if key in self.reader.map_dict:
                        if not torch.isnan(value):
                            self.reader.map_dict[key].append(value.numpy().tolist())

        return

    def calculate_score(self, result):
        """
        calculate the result comparing Model's Prediction with Ground Truth
        """
        print("[INFO] [Runner] [Calcuate Score]")

        score = 0
        avg_simillarity = 0 
        i = 0
        for k in self.reader.map_dict.keys():

            if self.reader.map_dict[k]:
                #print(torch.sum(torch.tensor(self.reader.map_dict[k])))
                pred_score = torch.mean(torch.tensor(self.reader.map_dict[k]))
                ground_truth = self.reader.ground_truth[k]
                #print(pred_score, ground_truth)
                avg_simillarity += pred_score
                score += (pred_score - ground_truth) ** 2
                i += 1

        print(i)
        print("[INFO] [Runner] [Avg_similarity]:", avg_simillarity / i)
        print("[INFO] [Runner] [Score]:", score / i)

        return score

    def run(self):
        """
        """

        # forward 
        #hiddens = self.forward()

        # similarity
        #result = self.calculate_similarity(hiddens)

        #for k,v in self.reader.map_dict.items():
            #print(k, v)
            #break

        # Score
        #score = self.calculate_score(result)
 
        #1.Predict
        scores = self.predict()

        #2.Estimate
        result = self.estimate(scores)

        #3.Analysis
        report = self.analysis()

        #4.Evaluate
        # metric = self.evaluate(scores)

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


