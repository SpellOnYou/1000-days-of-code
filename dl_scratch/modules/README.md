## Updates

- **2022.01.01**
  - resuscitate `runner` (which was killed at [09b_learner.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl2/09b_learner.ipynb)
  - Batchsize to 1
    - Remove N axis (i.e., `.squeeze(0)`) before training using callback
  - Callbacks
    - FramesizeFilter: raise batch exception error when n_frames are less than 100 after concatenating twice
    - SaveModelParam: save model checkpoint after one epoch
      - latest version: [commit](https://github.com/SpellOnYou/1000-days-of-code/commit/8d82ef3b96c88c0dac99bb50a5a8dcd83d77b892#diff-881f8a1768f15da891fe97569e1c0ddd037a2eb1c5ed66915e56d1d741688a19), `/gdrive/Shareddrives/Dion-Account/2122WS/8-dl4slp/coding-project/ser/checkpoints/ser-cnn-ffc-loss-bce_{n_epoch}.pt`
    - Recorder.tot_time: track time per epoch
  - Loss function
    - Changing loss function from cross_entropy to BCE_loss (i.e., single-label multi-class -> multi-label single class
    - [nn.BCELoss(), pytorch Binary Cross Entropy loss api](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    - changes in 1) labeling 2) final layer of model 3) calculating loss (maybe in runner) and others.
  - Normalization
    - Normalize with neutralized label (val 0, act 0), source: [Trantino et al., Section 3.4](https://publications.idiap.ch/attachments/papers/2019/Tarantino_INTERSPEECH_2021.pdf)


