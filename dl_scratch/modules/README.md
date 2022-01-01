## Updates

- **2022.01.01**
  - resuscitate `runner` (which was killed at [09b_learner.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl2/09b_learner.ipynb)
  - Batchsize to 1
    - Remove N axis (i.e., `.squeeze(0)`) before training using callback
  - Callbacks
    - FramesizeFilter: raise batch exception error when n_frames are less than 100 after concatenating twice
    - SaveModelParam: save model checkpoint after one epoch
    - Recorder.tot_time: track time per epoch
  - Loss function
    - Changing loss function from cross_entropy to BCE_loss (i.e., single-label multi-class -> multi-label single class
    - [nn.BCELoss(), pytorch Binary Cross Entropy loss api](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    - changes in 1) labeling 2) final layer of model 3) calculating loss (maybe in runner) and others.
  - Normalization
    - Normalize with neutralized label (val 0, act 0), source: [Trantino et al., Section 3.4](https://publications.idiap.ch/attachments/papers/2019/Tarantino_INTERSPEECH_2021.pdf)


