import torch  

class Score:
    def __init__(self):
        pass
    '''
    Args:
        predictions: Contain number of testcases, each has label name, dim: [num_testcases, num_answers]
        ground_truth: Contain number of testcases, each has label name, dim: [num_testcases, num_answers]
    '''
    
    def precision(self, predictions, ground_truth):
        precsision_scores = []
        for preds, gts in zip(predictions, ground_truth):
            preds = set(preds)
            gts = set(gts)
            if len(preds) == 0:
                precsision_scores.append(0.0)
            else:
                precsision_scores.append(len(preds.intersection(gts))/len(preds))
        
        return sum(precsision_scores)/len(precsision_scores)
    
    def recall(self, predictions, ground_truth):
        recall_scores = []
        for preds, gts in zip(predictions, ground_truth):
            preds = set(preds)
            gts = set(gts)
            if len(gts) == 0:
                recall_scores.append(0.0)
            else:
                recall_scores.append(len(preds.intersection(gts))/len(gts))

        return sum(recall_scores)/len(recall_scores)
    
    def f1(self, predictions, ground_truth):
        prec = self.precision(predictions, ground_truth)
        rec = self.recall(predictions, ground_truth)
        if prec + rec == 0:
            return 0.0
        
        return 2*prec*rec/(prec + rec)
    