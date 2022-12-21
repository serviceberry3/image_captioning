'''
Tokenizer is a compact pure-Python (>= 3.6) executable program and module for tokenizing Icelandic text. 
It converts input text to streams of tokens, where each token is a separate word, punctuation sign, number/amount, date, 
e-mail, URL/URI, etc. It also segments the token stream into sentences, considering corner cases such as abbreviations and 
dates in the middle of sentences.
'''
from .tokenizer.ptbtokenizer import PTBTokenizer

'''
BLEU scores are used to evaluate accuracy of nn-generated captions compared to ground truth captions
'''
from .bleu.bleu import Bleu


from .meteor.meteor import Meteor
from rouge.rouge import Rouge
from .cider.cider import Cider

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes

        self.params = {'image_id': coco.getImgIds()}


    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()

        #empty dicts to hold ground truth captions and network-result captions for each image id that was run as val data
        gts = {}
        res = {}

        for imgId in imgIds:
            if self.coco.config.dataset == 'coco':
                #CHANGED BY NWEINER 12/19 -- use new coco dict names
                #gts[imgId] = self.coco.imgToAnns[imgId]
                gts[imgId] = self.coco.imgId_to_ann[imgId]

                #res[imgId] = self.cocoRes.imgToAnns[imgId]
                res[imgId] = self.cocoRes.imgId_to_ann[imgId]
            elif self.coco.config.dataset == 'sbu':
                gts[imgId] = self.coco.imgId_to_cap[imgId]

                #res[imgId] = self.cocoRes.imgToAnns[imgId]
                res[imgId] = self.cocoRes.imgId_to_ann[imgId]


        #print("gts is", gts)
        #print("res is", res)

        # =================================================
        # Set up scorers
        # =================================================
        print('Tokenization of the ground truth and network-run dicts...')
        tokenizer = PTBTokenizer()


        if self.coco.config.dataset == 'coco':
            #tokenize the two dicts
            gts = tokenizer.tokenize(gts)

            res = tokenizer.tokenize(res)
        elif self.coco.config.dataset == 'sbu':
            #tokenize the two dicts
            gts = tokenizer.tokenize_sbu(gts, imgIds)

            res = tokenizer.tokenize_sbu(res, imgIds)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')

        #different scoring metrics
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(),"METEOR"),
            #(Rouge(), "ROUGE_L"),
            #(Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores using each of 4 scoring methods
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))

            #compare generated caption to ground truth
            score, scores = scorer.compute_score(gts, res)

            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))

                
        self.setEvalImgs()



    def setEval(self, score, method):
        self.eval[method] = score



    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score


    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]