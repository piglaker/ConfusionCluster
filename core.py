

import time




def _get_metrics(training_args=None):
    """
    #https://huggingface.co/metrics
    #accuracy,bertscore, bleu, bleurt, coval, gleu, glue, meteor,
    #rouge, sacrebleu, seqeval, squad, squad_v2, xnli
    metric = load_metric() 
    """
    
    import numpy as np
    from datasets import load_metric
    from tqdm import tqdm
    def compute_metrics(eval_preds):
        """
        reference: https://github.com/ACL2020SpellGCN/SpellGCN/blob/master/run_spellgcn.py
        """
        print("Compute Metrics for Aligned CSC")
        Achilles = time.time()

        sources, preds, labels = eval_preds# (num, length) np.array
 
        tp, fp, fn = 0, 0, 0

        sent_p, sent_n = 0, 0

        for i in tqdm(range(len(sources))):
            #print(sources[i])
            #print(preds[i])
            #print(labels[i])

            source, pred, label = np.array(sources[i]), np.array(preds[i]), np.array(labels[i])

            #print(source, preds, label)

            source, label = source[ source != -100], label[label != -100]

            source, label = source[source != 0],  label[label != 0]#pad idx for input_ids 

            #we guess pretrain Masked Language Model bert lack the surpvised sighan for 101 & 102 ( [CLS] & [SEP] ) , so we just ignore
            #source, pred, label = np.where(source == 102, 101, source), np.where(pred == 102, 101, pred), np.where(label == 102, 101, label) 

            source, label = np.where(source == 102, 101, source), np.where(label == 102, 101, label)

            source, label = source[ source != 101 ], label[ label != 101]

            pred = pred[1:]

            pred = np.where(pred == 102, 101, pred)

            pred = pred[ pred != 101 ]

            source = source[:len(label)]
            pred = pred[:len(label)]

            pred = np.concatenate((pred, np.array([ 0 for i in range(len(label) - len(pred))])), axis=0)

            if len(pred) != len(source) or len(label) != len(source):
                print("Warning : something goes wrong when compute metrics, check codes now.")
                print(len(source), len(pred), len(label))
                print("source: ", source)
                print("pred: ", pred)
                print("label:", label)
                print("raw source: ", sources[i])
                print("raw pred: ", preds[i])
                print("raw label:", labels[i])
                exit()
            try:
                (pred != source).any()
            except:
                print(pred, source)
                print(" Error Exit ")
                exit(0)

            #if i < 5:
            #print(source)
            #print(pred)
            #print(label)
            #print((pred != source).any())
            #print((pred == label).all())
            #print((label != source).any())


            if not training_args:
                # label: [101, 2,... 3, 102]
                if (pred != source).any():
                    sent_p += 1
                    #print("sent_p")
                    if (pred == label).all():
                        tp += 1
                        # print("tp")

                if (label != source).any():
                    sent_n += 1
                    #print("sent_n")
            else:
                # label : [ 1,1,1,1,1 ]
                if (pred != 1).any():
                    sent_p += 1

                    if (pred == label).all():
                        tp += 1
            
                if (label != 1).any():
                    sent_n += 1

        #print(tp, sent_p, sent_n)

        precision = tp / (sent_p + 1e-10)

        recall = tp / (sent_n + 1e-10)

        F1_score = 2 * precision * recall / (precision + recall + 1e-10)

        Turtle = time.time() - Achilles

        if F1_score < 0.05:
            print("Warning : metric score is too Low , maybe something goes wrong, check your codes please.")
            #exit(0)
        return {"F1_score": float(F1_score), "Precision":float(precision),  "Recall":float(recall),"Metric_time":Turtle}

    return compute_metrics