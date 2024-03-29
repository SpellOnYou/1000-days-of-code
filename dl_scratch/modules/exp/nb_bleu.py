
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/bleu_metric.ipynb



pred = [1,2,3,7,5,1,1]


def __init__(self, ngram, max_n=5000): self.ngram,self.max_n = ngram,max_n
def __eq__(self, other):
    if len(self.ngram) != len(other.ngram): return False
    return np.all(np.array(self.ngram) == np.array(other.ngram))
def __hash__(self): return int(sum([o * self.max_n**i for i,o in enumerate(self.ngram)]))

return x if n==1 else [NGram(x[i:i+n], max_n=max_n) for i in range(len(x)-n+1)]

pred_grams,targ_grams = get_grams(pred, n, max_n=max_n),get_grams(targ, n, max_n=max_n)
pred_cnt,targ_cnt = Counter(pred_grams),Counter(targ_grams)
return sum([min(c, targ_cnt[g]) for g,c in pred_cnt.items()]),len(pred_grams)

corrects = [get_correct_ngrams(pred,targ,n,max_n=max_n) for n in range(1,5)]
n_precs = [c/t for c,t in corrects]
len_penalty = exp(1 - len(targ)/len(pred)) if len(pred) < len(targ) else 1
return len_penalty * ((n_precs[0]*n_precs[1]*n_precs[2]*n_precs[3]) ** 0.25)

pred_len,targ_len,n_precs,counts = 0,0,[0]*4,[0]*4
for pred,targ in zip(preds,targs):
    pred_len += len(pred)
    targ_len += len(targ)
    for i in range(4):
        c,t = ngram_corrects(pred, targ, i+1, max_n=max_n)
        n_precs[i] += c
        counts[i] += t
n_precs = [c/t for c,t in zip(n_precs,counts)]
len_penalty = exp(1 - targ_len/pred_len) if pred_len < targ_len else 1
return len_penalty * ((n_precs[0]*n_precs[1]*n_precs[2]*n_precs[3]) ** 0.25)

def __init__(self, vocab_sz):
    self.vocab_sz = vocab_sz
    self.name = 'bleu'

def on_epoch_begin(self, **kwargs):
    self.pred_len,self.targ_len,self.n_precs,self.counts = 0,0,[0]*4,[0]*4

def on_batch_end(self, last_output, last_target, **kwargs):
    last_output = last_output.argmax(dim=-1)
    for pred,targ in zip(last_output.cpu().numpy(),last_target.cpu().numpy()):
        self.pred_len += len(pred)
        self.targ_len += len(targ)
        for i in range(4):
            c,t = get_correct_ngrams(pred, targ, i+1, max_n=self.vocab_sz)
            self.n_precs[i] += c
            self.counts[i] += t

def on_epoch_end(self, last_metrics, **kwargs):
    n_precs = [c/t for c,t in zip(n_precs,counts)]
    len_penalty = exp(1 - targ_len/pred_len) if pred_len < targ_len else 1
    bleu = len_penalty * ((n_precs[0]*n_precs[1]*n_precs[2]*n_precs[3]) ** 0.25)
    return add_metrics(last_metrics, bleu)