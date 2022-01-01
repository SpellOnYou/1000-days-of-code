"""How to save result prediciton to submission format?"""

from collections import defaultdict
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
ser_path = Path('/gdrive/Shareddrives/Dion-Account/2122WS/8-dl4slp/coding-project/ser/')

subm = defaultdict(str)
for k, v in res_dict.items():
    subm[k] = {}
    subm[k]['activation'] = v[0]
    subm[k]['valence'] = v[1]


with open(ser_path/f'uploads/xresent50-20epochs-f1:0.69-{my_serial}.json', 'w') as f:
    json.dump(subm, f)


learn = cnn_learner(model, data, loss_func, opt_func)

learn.model.torch.save(path)
learn.model.load_state_dict(path)



# ------------------------------------------------

"""How to get same length of frames using global pooing?"""

drive_data_path = ser_path/'data'
data_path = Path('/content/data'); data_path.mkdir()

!tar -xf {drive_data_path/'ser-data-v2-1.tar.gz'} -C {data_path}

features = [torch.load(path).t() for path in (data_path/'ser/dev').ls()[:100]] # (n_logmel, frames) each element will have differnt shape on 2nd dim

t1= features[0].unsqueeze(1); t.shape
# torch.Size([26, 1, 105])

model = nn.Sequential(
    nn.Conv1d(1,8, kernel_size=20, stride=1),
    nn.Conv1d(8,16, kernel_size=15),
    nn.Conv1d(16,32, kernel_size=7),
    nn.Conv1d(32,64, kernel_size=5),
    nn.Conv1d(64,128, kernel_size=3),
    nn.Conv1d(128,128, kernel_size=3), #26, 128, 28
    nn.AdaptiveAvgPool2d((128, 1)), #26, 128, 1
    )

"""
for idx, inst in enumerate(features):
    out = model(inst.unsqueeze(1))
    assert out.shape == one_out.shape, f"{idx} data has shape of {inst.shape}"
"""
# torch.Size([1, 26, 128, 28]) # note: 128 comes from channel (of frames), 28 is remained(?) features

# x.shape >>>torch.Size([26, 128, 1])

"""dev json legnth
[(72, 54),
 (43, 55),
 (5, 58),
 (76, 72),
 (47, 75),
 (60, 77),
 (44, 80),
 (14, 82),
 (92, 82),
 (48, 84)]
"""

#export
#################################################
### TUNING Callback functionds ###
#################################################



learn = cnn_learner(xresnet34, data, loss_func, opt_func)
learn.fit(5, cbs=cbscheds)
class RNNTrainer(Callback):
    def __init__(self, α, β): self.α,self.β = α,β
    
    def after_pred(self):
        #Save the extra outputs for later and only returns the true output.
        self.raw_out,self.out = self.pred[1],self.pred[2]
        self.run.pred = self.pred[0]
    
    def after_loss(self):
        #AR and TAR
        if self.α != 0.:  self.run.loss += self.α * self.out[-1].float().pow(2).mean()
        if self.β != 0.:
            h = self.raw_out[-1]
            if h.size(1)>1: self.run.loss += self.β * (h[:,1:] - h[:,:-1]).float().pow(2).mean()
                
    def begin_epoch(self):
        #Shuffle the texts at the beginning of the epoch
        if hasattr(self.dl.dataset, "batchify"): self.dl.dataset.batchify()
def cross_entropy_flat(input, target):
    bs,sl = target.size()
    return F.cross_entropy(input.view(bs * sl, -1), target.view(bs * sl))

def accuracy_flat(input, target):
    bs,sl = target.size()
    return accuracy(input.view(bs * sl, -1), target.view(bs * sl))

cbs = [partial(AvgStatsCallback,accuracy_flat),
       CudaCallback, Recorder,
       partial(GradientClipping, clip=0.1),
       partial(RNNTrainer, α=2., β=1.),
       ProgressCallback]

learn = Learner(model, data, cross_entropy_flat, lr=5e-3, cb_funcs=cbs, opt_func=adam_opt())


def cnn_learner(arch, data, loss_func, opt_func, c_in=None, c_out=None,
                lr=1e-2, cuda=True, norm=None, progress=True, mixup=0, xtra_cb=None, **kwargs):
    cbfs = [partial(AvgStatsCallback,accuracy)]+listify(xtra_cb)
    if progress: cbfs.append(ProgressCallback)
    if cuda:     cbfs.append(CudaCallback)
    if norm:     cbfs.append(partial(BatchTransformXCallback, norm))
    if mixup:    cbfs.append(partial(MixUp, mixup))
    arch_args = {}
    if not c_in : c_in  = data.c_in
    if not c_out: c_out = data.c_out
    if c_in:  arch_args['c_in' ]=c_in
    if c_out: arch_args['c_out']=c_out
    return Learner(arch(**arch_args), data, loss_func, opt_func=opt_func, lr=lr, cb_funcs=cbfs, **kwargs)

class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9
        
    def begin_batch(self): 
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups: pg['lr'] = lr
            
    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss

class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses = [],[]

    def after_batch(self):
        if not self.in_train: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self): plt.plot(self.lrs)
    def plot_loss(self): plt.plot(self.losses)

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])



# ---------


from google.colab import drive
drive.mount('/gdrive')

# upload local files in exp
from exp.nb_12a import *

root_path = Path('/gdrive/Shareddrives/Dion-Account/2122WS/8-dl4slp/coding-project/ser'); root_path.ls()

dev_path = (path2/'data/v1/dev')
# audios = get_files(dev_path)
class AudioList(ItemList):
    @classmethod
    def from_files(cls, path, extensions = None, recurse=True, include=None, **kwargs):
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
    
    def get(self, fn):
        return torch.load(fn)
        
al=AudioList.from_files(dev_path, tfms=tfms)

testset=torch.cat([al[idx] for idx, _ in enumerate(al.items)], dim=0)
testset.shape

def get_predictions(learn, dev):
    test_n = dev.shape[0]
    learn.model.eval()
    res = []
    with torch.no_grad():        
        for i in range((test_n-1)//bs + 1):
            xb = dev[i*bs:(i+1)*bs]
            out = learn.model(xb)
            res += [o.item() for o in out.argmax(1)]
    return res


res = get_predictions(learn, testset.unsqueeze(1))



class Reshape():
    _order=12
    def __call__(self, item):
        w, h = item.shape
        return item.view(h, w)
# Mutants of input tensor
class PadorTrim():
    
    _order = 20
    def __init__(self, max_len):
        self.max_len = max_len
    def __call__(self, ad):
        # h - logmel, here 27, w - frames / various
        h, w = ad.shape
        pad_size = self.max_len - w
        if pad_size >0: return torch.cat((ad, torch.zeros(h, pad_size).to(ad.device)), dim=1)
        else: return ad[:, :self.max_len]
class DummyChannel():
    _order = 30
    def __call__(self, item):
        return item.unsqueeze(0)
class SpecAugment():
    _order=99
    def __init__(self, max_mask_pct=0.2, freq_masks=1, time_masks=1, replace_with_zero=False):
        self.max_mask_pct, self.freq_masks, self.time_masks, self.replace_with_zero = \
        max_mask_pct, freq_masks, time_masks, replace_with_zero
        if not 0 <= self.max_mask_pct <= 1.0: 
            raise ValueError( f"max_mask_pct must be between 0.0 and 1.0, but it's {self.max_mask_pct}")

    def __call__(self, spec):
        _, n_mels, n_steps = spec.shape
        F = math.ceil(n_mels * self.max_mask_pct) # rounding up in case of small %
        T = math.ceil(n_steps * self.max_mask_pct)
        fill = 0 if self.replace_with_zero else spec.mean()
        for i in range(0, self.freq_masks):
            f = random.randint(0, F)
            f0 = random.randint(0, n_mels-f)
            spec[0][f0:f0+f] = fill
        for i in range(0, self.time_masks):
            t = random.randint(0, T)
            t0 = random.randint(0, n_steps-t)
            spec[0][:,t0:t0+t] = fill
        return spec

masker = SpecAugment(freq_masks=2, time_masks=2, max_mask_pct=0.1)

tfms = [Reshape(), PadorTrim(250), DummyChannel(), masker]

al=AudioList.from_files(train_path, tfms=tfms)

def re_labeler(fn, pat, subcl='act'):
    assert subcl in ['act', 'val', 'all']
    if subcl=='all': return ''.join(re.findall(pat, str(fn)))
    else:
        return re.findall(pat, str(fn))[0] if pat == 'act' else re.findall(pat, str(fn))[1]

label_pat = r'_(\d+)'
emotion_labeler = partial(re_labeler, pat=label_pat, subcl='all')
sd = SplitData.split_by_func(al, partial(random_splitter, p_valid=0.2))
ll = label_by_func(sd, emotion_labeler, proc_y=CategoryProcessor())
bs=64
c_in = ll.train[0][0].shape[0]
c_out = len(uniqueify(ll.train.y))
data = ll.to_databunch(bs,c_in=c_in,c_out=c_out)

opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-1, )
loss_func = LabelSmoothingCrossEntropy()
learn = cnn_learner(xresnet50, data, loss_func, opt_func)

from sklearn.metrics import f1_score

def multi_f1score(input, target):
    with torch.no_grad():
        label_convert = {0:[1, 1], 1:[0, 0], 2:[1,0], 3:[0, 1]}
        pred_np = np.array(list(map(lambda o: label_convert[int(o)], input.argmax(1))))
        targ_np = np.array(list(map(lambda o: label_convert[int(o)], target)))
        return f1_score(targ_np[:, 0], pred_np[:, 0]), f1_score(targ_np[:, 1], pred_np[:, 1])
