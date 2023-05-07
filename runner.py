
import pickle

from collections.abc import Mapping
from numpy import average, source

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


from tqdm import tqdm

from similar_score import calcuate

from transformers import DataCollatorWithPadding

class BaseRunner():
    def __init__():
        
        return

class ConfusionClusterRunner():
    def __init__(self, model, args, reader, confusion_reader,  data_collator, topk=5):
        """
        """
        self.args = args
        
        self.dataset = None
        
        self.cluster_algorithm = None
        
        self.model = model
        
        self.confusionset = None
        
        self.reader = reader

        self.confusion_reader = self.reader.confusion_reader

        #self.confusion_reader.run()

        self.data_collator = data_collator

        self.device = 'cuda'

        self.batch_size = 64

        self.all_result = None

        self.topk = topk

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
        #print(BertOutput)
        #print(torch.nan_to_num(BertOutput.logits[0][0]))
        #exit()

        return torch.softmax(BertOutput.logits, 2).detach().cpu()

    def forward(self):
        """
        """
        print("[INFO] [Runner] [Forward]")
        encodings = self.reader.get_dataset()
        source_set, target_set, confusion_set = encodings["source"], encodings["target"], encodings["confusion"]

        if self.reader.is_ReaLiSe:
            def trans2mydataset(some_set):
                new = []
                for feature in some_set:
                    tmp = {}
                    tmp["input_ids"] = feature["src_idx"][:128]
                    tmp["labels"] = feature["tgt_idx"][:128]
                    tmp["attention_mask"] = ([1] * len(tmp["input_ids"]))[:128]#feature["lengths"])[:128]
                    new.append(tmp)
        
                return new

            source_set, target_set, confusion_set = trans2mydataset(source_set), trans2mydataset(target_set), trans2mydataset(confusion_set)

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

        confusion_dataloader = DataLoader(
            confusion_set,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

        source_host = torch.tensor([])

        for inputs in tqdm(source_dataloader ):
            inputs = self._prepare_inputs(inputs)
            hiddens = self.single_step(inputs)
            source_host = torch.cat( (source_host, hiddens), dim=0)

        target_host = torch.tensor([])

        for inputs in tqdm( target_dataloader ):
            inputs = self._prepare_inputs(inputs)
            hiddens = self.single_step(inputs)
            target_host = torch.cat( (target_host, hiddens), dim=0)

        confusion_host = torch.tensor([])

        for inputs in tqdm( confusion_dataloader ):
            inputs = self._prepare_inputs(inputs)
            hiddens = self.single_step(inputs)
            confusion_host = torch.cat( (confusion_host, hiddens), dim=0)

        return source_host, target_host, confusion_host

    def predict(self):
        """
        """

        print("[INFO] [Runner] [Predict]")

        if isinstance(self.model, str):
            # we hack
            print("[INFO] [Runner] [Hack]")
            path_1 = "./models/"+self.model+"/source_logits.pth" 
            #with open(path_1 , "rb") as f:
            #    source_host = pickle.load(f)
            source_host = torch.load(path_1)

            #with open("./models/"+self.model+"/masked_logits.pkl" , "rb") as f:
            #    masked_host = pickle.load(f)  
            
            path_2 = "./models/"+self.model+"/masked_logits.pth"

            masked_host = torch.load(path_2)

            return source_host, masked_host

        encodings = self.reader.get_dataset()
        source_set, target_set, masked_set = encodings["source"], encodings["target"], encodings["masked"]

        if self.reader.is_ReaLiSe:
            def trans2mydataset(some_set):
                new = []
                for feature in some_set:
                    tmp = {}
                    tmp["input_ids"] = feature["src_idx"][:128]
                    tmp["labels"] = feature["tgt_idx"][:128]
                    tmp["attention_mask"] = ([1] * len(tmp["input_ids"]))[:128]#feature["lengths"])[:128]
                    new.append(tmp)
        
                return new

            source_set, masked_set = trans2mydataset(source_set), trans2mydataset(masked_set)

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
            #print(scores)
            #exit()
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
            source, target = self.reader.encoding["source"][i][self.reader.input_token], self.reader.encoding["source"][i][self.reader.label_token]#self.reader.encoding["target"][i][self.reader.input_token]
            
            score = source_scores[i]
            masked_score = masked_scores[i]

            result_host = []

            for j in range(len(source)):
                x = source[j]
                y = target[j]

                if x != y:

                    noise = None

                    all_possible_noise = masked_score.topk(20, dim=1)[-1][j]

                    if self.reader.tokenizer.decode(x) in self.confusion_reader.confusion:
                        confusion_x = self.confusion_reader.confusion[self.reader.tokenizer.decode(x)]
                    else:
                        confusion_x = []
                    
                    # noise select
                    
                    for possible_noise in all_possible_noise:
                        if possible_noise != x and possible_noise != y and self.reader.tokenizer.decode(possible_noise) not in confusion_x:                   
                            noise = possible_noise
                            break
                    
                    #if not noise:
                    #    continue

                    p_y_cx = score[j][y]

                    p_noise_cx = score[j][noise]

                    p_y_cm = masked_score[j][y]

                    p_noise_cm = masked_score[j][noise]

                    #print(p_y_cx, p_noise_cx, p_y_cm, p_noise_cm)

                    result = {
                                "C":self.reader.tokenizer.decode(source),
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
                                "e": p_y_cm / ( 1 - masked_score[j].max() + 1e-10 ) , 
                    }

                    result_host.append(result)

            all_result.append(result_host)

        self.all_result = all_result

        return all_result

    def analysis(self):
        """
        """
        print("[INFO] [Runner] [Analysis]")
        count = 0
        mlm_count = 0
        correct_count = 0

        baseline_e = 0

        for example in self.all_result:
            for result in example:
                bad_mlm = result["p_noise_cm"] > result["p_y_cm"]
                good_correction = result["p_y_cx"] > result["p_noise_cx"]

                if bad_mlm:
                    baseline_e += result["e"]
                    #print(result)
                if  bad_mlm and good_correction :
                    count += 1
                if bad_mlm:
                    mlm_count += 1

                if good_correction:
                    correct_count += 1

        length =  sum( [ len(i) for i in self.all_result] )

        print(
            "[inter/mlm]", count / mlm_count, \
            "[baseline]", baseline_e / mlm_count, \
            "[Bad_MLM and Good_Correction]:", count / length, \
            "[Bad_MLM]:", mlm_count / length, \
            "[Good_Correction]:", correct_count / length, \
        )



    def evaluate(self, scores):
        """
        """
        print("[INFO] [Runner] [Evaluate]")

        source_scores, masked_scores = scores

        print("TopK:", self.topk)

        preds = source_scores[2062:]#[ torch.argmax(s, dim=-1) for s in source_scores[1062:] ]
        masked_preds =  masked_scores[2062:]#[ torch.argmax(s, dim=-1) for s in masked_scores[1062:] ]
        sources = [ i[self.reader.input_token] for i in self.reader.encoding["source"] ][2062:]
        masked = [ i[self.reader.input_token] for i in self.reader.encoding["masked"] ][2062:] 
        labels = [ i[self.reader.label_token] for i in self.reader.encoding["source"] ][2062:]

        preds13 = source_scores[:1000]
        masked_preds13 = masked_scores[:1000]
        sources13 = [ i[self.reader.input_token] for i in self.reader.encoding["source"] ][:1000]
        masked13 = [ i[self.reader.input_token] for i in self.reader.encoding["masked"] ][:1000] 
        labels13 = [ i[self.reader.label_token] for i in self.reader.encoding["source"] ][:1000]         

        preds14 = source_scores[1000:2062]#[ torch.argmax(s, dim=-1) for s in source_scores[:1062] ] 
        masked_preds14 =  masked_scores[1000:2062]#[ torch.argmax(s, dim=-1) for s in masked_scores[:1062] ] 
        sources14 = [ i[self.reader.input_token] for i in self.reader.encoding["source"] ][1000:2062] 
        masked14 = [ i[self.reader.input_token] for i in self.reader.encoding["masked"] ][1000:2062]         
        labels14 = [ i[self.reader.label_token] for i in self.reader.encoding["source"] ][1000:2062] 
        
        # print(sources[0])
        # print(preds[0])
        # print(masked_preds[0])
        # print(labels[0])

        #print(self.reader.tokenizer.decode(sources[1]))
        #print(self.reader.tokenizer.decode(preds[1]))
        #print(self.reader.tokenizer.decode(masked_preds[1]))
        #print(self.reader.tokenizer.decode(labels[1]))
        #exit()
        
        print("{13:")
        self.metric((sources13, preds13, labels13), need_postpre=True)
        print("{13 mask:")
        self.metric((masked13, masked_preds13, labels13), need_postpre=True)

        print("{14:")
        self.metric((sources14, preds14, labels14))
        print("{14 mask:")
        self.metric((masked14, masked_preds14, labels14))
        
        print("{15")
        self.metric((sources, preds, labels))
        print("{15 mask")
        self.metric((masked, masked_preds, labels))

        return

    def get_sim(self, i):
        confusion = self.confusion_reader[self.reader.tokenizer.decode(i)]
        return [ self.reader.tokenizer.convert_tokens_to_ids(j) for j in confusion]

    def metric(self, input, need_postpre=False):
        """
        """
        sources, preds, labels = input

        tp, sent_p, sent_n = 0, 0, 0
        d_tp, d_sent_p, d_sent_n = 0, 0, 0

        pair = {}

        for j in range(len(sources)):
            #得，地，的 4638, 1765, 2533
            logits = torch.softmax(preds[j], dim=1)

            topk = torch.topk(logits, self.topk)[-1]

            src, topk, label = torch.tensor(sources[j]), topk, torch.tensor(labels[j])
                    
            #if self.model.find("ReaLiSe") >= 0:
            #    tmp = tmp[1:label.shape[0]-1, :]
            #    src = src[1:label.shape[0]-1]
            #    label = label[1:label.shape[0]-1]

            tmp = topk[:, 0]

            def cls2sep(x):
                return torch.where(x != 101, x , 102)

            src, tmp, label = cls2sep(src), cls2sep(tmp), cls2sep(label) 

            def postpre(x):
                x = torch.where(x != 4638, x, 2533)
                x = torch.where(x != 1765, x, 2533)
                return x
            
            if need_postpre:
                src, tmp, label = postpre(src), postpre(tmp), postpre(label)

            #print(src)
            #print(tmp)
            #print(label)
            #print(self.reader.tokenizer.decode(src))
            #print(self.reader.tokenizer.decode([i.item() for i in tmp]))
            #print(self.reader.tokenizer.decode(label)) 
            #exit()

            ### ReaLiSe is Shit
            # It change the x and output
            
            src = src[ src != -100]
            src = src[ src != 0]
            label = label[ label != 0]
            label = label[ label != -100] 
            tmp = tmp[:label.shape[0]]

            pred = tmp

            #print(pred, src, label)
            #print(pred.shape, tmp.shape, src.shape, label.shape)
            #exit()
            res = (tmp == label.unsqueeze(-1)).sum(dim=1).sum()

            if ( pred != src ).any() :
                sent_p += 1
                d_sent_p += 1
                #if (res == label.shape[0]):
                
                if ((src != label).int() == (src != pred).int()).all():
                    d_tp += 1
                
                if self.topk > 1:
                    new_pred = []
                    for i in range(len(src)):
                        if pred[i] != src[i]:
                            confusion = self.get_sim(src[i])
                            for j in range(len(topk[i])):
                                if topk[i][j] in confusion:
                                    new_pred.append(topk[i][j])
                                    break
                            else:
                                new_pred.append(pred[i])
                        else:
                            new_pred.append(pred[i])
                    if (torch.tensor(new_pred) == label).all():
                        tp += 1 
                else:
                    if (pred == label).all(): 
                        tp += 1

            if ( src != label).any():
                sent_n += 1
                d_sent_n += 1

        d_p = d_tp / (d_sent_p + 1e-10)
        d_r = d_tp / (d_sent_n + 1e-10)
        d_f1 = 2 * d_p * d_r / (d_p + d_r + 1e-10)

        precision = tp / (sent_p + 1e-10)
        recall = tp / (sent_n + 1e-10)
        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        print("pre: ", precision, "recall: ", recall, "F1: ", F1_score, "d_pre:", d_p, "d_recall:", d_r, "d_f1:",d_f1)

        return 

    def calculate_similarity(self, hiddens):
        """
        calculate Model's Prediction
        """
        print("[INFO] [Runner] [Calculate_Similarity] ")

        source_hiddens, target_hiddens, confusion_hiddens = hiddens
        # Euclidean normalize:https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

        #print(torch.tensor(source_hiddens).shape)

        #print(torch.tensor(target_hiddens).shape)
        #exit()
        source_hiddens, target_hiddens, confusion_hiddens = torch.nn.functional.normalize(source_hiddens, dim=1), \
            torch.nn.functional.normalize(target_hiddens,dim=1), \
                torch.nn.functional.normalize(confusion_hiddens,dim=1) 

        cos = nn.CosineSimilarity(dim=0, eps=1e-16)

        l_align, l_uniform = 0, 0

        for i in tqdm(range(len(self.reader.data))):
            source, target = self.reader.encoding["source"][i][self.reader.input_token], self.reader.encoding["target"][i][self.reader.input_token]

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
                     
                    l_align += (a - b).pow(2).mean()
                    #l_uniform += (a - b).pow(2).mul(-2).exp().mean().log()
                    
                    #print(value, torch.isnan(value)) 
                    if key in self.reader.map_dict:
                        if not torch.isnan(value):
                            self.reader.map_dict[key].append(value.numpy().tolist())
                
                    #a = source_hiddens[i][j]
                    y = confusion_hiddens[i]
                    import random
                    #r = random.choice( [ i for i in range(len(y)) if i != j ] )
                    b = y[j]
                    l_uniform += ( a - b).pow(2).mul(-2).exp().mean().log()

        print("[Contrastive] l_align: ", l_align.mean())
        print("[Contrastive] l_uniform:", l_uniform.mean())

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
        hiddens = self.forward()

        # similarity
        result = self.calculate_similarity(hiddens)

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
        metric = self.evaluate(scores)

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


