import torch
import torchvision
import fastai
from fastai.vision import Learner, DatasetType, flip_lr, progress_bar
from TOOLS.mask_functions import mask2rle
from utils import seed_everything
from models.fastai_unet_learner import (
    acc_create_opt,
    unet_learner,
    dice,
    set_BN_momentum,
    AccumulateStep,
    dice_overall,
)
from dataset.fastai_data import new_transform, get_data
import numpy as np
import PIL
import pandas as pd
import gc
from functools import partial
from pathlib import Path

sz = 256
bs = 8
n_acc = 64 // bs  # gradinet accumulation steps
nfolds = 4
SEED = 2019
circle_1 = 1
circle_2 = 1
lr_0 = 1e-3

# eliminate all predictions with a few (noise_th) pixesls
noise_th = 75.0 * (sz / 128.0) ** 2  # threshold for the number of predicted pixels
best_thr0 = 0.2  # preliminary value of the threshold for metric calculation

IMAGE_STATS_DICT = {
    128: ([0.615, 0.615, 0.615], [0.291, 0.291, 0.291]),
    256: ([0.540, 0.540, 0.540], [0.264, 0.264, 0.264]),
    512: ([0.529, 0.529, 0.529], [0.259, 0.259, 0.259]),
}


TRAIN = f"input/train_{sz}"
TEST = f"input/test_{sz}"
MASKS = f"input/masks_{sz}"
backbone = torchvision.models.resnet101
model_name = "resnet101"
checkpoint_path = f"checkpoints/fastai-{model_name}/"
device = [0]
Path(checkpoint_path).mkdir(exist_ok=True)
stats = IMAGE_STATS_DICT[sz]


# Prediction with flip TTA
def pred_with_flip(
    learn: fastai.basic_train.Learner,
    ds_type: fastai.basic_data.DatasetType = DatasetType.Valid,
):
    # get prediction
    preds, ys = learn.get_preds(ds_type)
    preds = preds[:, 1, ...]
    # add fiip to dataset and get prediction
    learn.data.dl(ds_type).dl.dataset.tfms.append(flip_lr())
    preds_lr, ys = learn.get_preds(ds_type)
    del learn.data.dl(ds_type).dl.dataset.tfms[-1]
    preds_lr = preds_lr[:, 1, ...]
    ys = ys.squeeze()
    preds = 0.5 * (preds + torch.flip(preds_lr, [-1]))
    del preds_lr
    gc.collect()
    torch.cuda.empty_cache()
    return preds, ys


seed_everything(SEED)

Learner.create_opt = acc_create_opt

fastai.data_block.ItemLists.transform = new_transform

get_data_p = partial(get_data, TRAIN, TEST, stats, sz, bs)
dice_p = partial(dice, best_thr0, noise_th)
# ### Training

# Expand the following cell to see the model printout. The model is based on Unet like architecture with ResNet34 based pretrained encoder. The upscaling is based on [pixel shuffling technique](https://arxiv.org/pdf/1609.05158.pdf). On the top, hypercolumns are added to provide additional skip-connections between the upscaling blocks and the output.

scores, best_thrs = [], []

for fold in range(nfolds):
    print("fold: ", fold)
    data = get_data_p(fold)
    learn = unet_learner(data, backbone, metrics=[dice_p])
    if len(device) > 1:
        learn.model = torch.nn.DataParallel(learn.model, device_ids=device)
    learn.clip_grad(1.0)
    set_BN_momentum(learn.model)

    # fit the decoder part of the model keeping the encode frozen
    lr = lr_0
    learn.fit_one_cycle(circle_1, lr, callbacks=[AccumulateStep(learn, n_acc)])

    # fit entire model with saving on the best epoch
    learn.unfreeze()
    learn.fit_one_cycle(
        circle_2, slice(lr / 80, lr / 2), callbacks=[AccumulateStep(learn, n_acc)]
    )
    learn.save(checkpoint_path + "/fold" + str(fold))

    # prediction on val and test sets
    preds, ys = pred_with_flip(learn)
    pt, _ = pred_with_flip(learn, DatasetType.Test)

    if fold == 0:
        preds_test = pt
    else:
        preds_test += pt

    # convert predictions to byte type and save
    preds_save = (preds * 255.0).byte()
    torch.save(preds_save, checkpoint_path + "/preds_fold" + str(fold) + ".pt")
    np.save(checkpoint_path + "/items_fold" + str(fold), data.valid_ds.items)

    # remove noise
    preds[preds.view(preds.shape[0], -1).sum(-1) < noise_th, ...] = 0.0

    # optimal threshold
    # The best way would be collecting all oof predictions followed by a single threshold
    # calculation. However, it requres too much RAM for high image resolution
    dices = []
    thrs = np.arange(0.01, 1, 0.01)
    for th in progress_bar(thrs):
        preds_m = (preds > th).long()
        dices.append(dice_overall(preds_m, ys).mean())
    dices = np.array(dices)
    scores.append(dices.max())
    best_thrs.append(thrs[dices.argmax()])

    if fold != nfolds - 1:
        del preds, ys, preds_save
    gc.collect()
    torch.cuda.empty_cache()

preds_test /= nfolds

print("scores: ", scores)
print("mean score: ", np.array(scores).mean())
print("thresholds: ", best_thrs)
best_thr = np.array(best_thrs).mean()
print("best threshold: ", best_thr)


# convert predictions to byte type and save
preds_save = (preds_test * 255.0).byte()
torch.save(preds_save, checkpoint_path + "/preds_test.pt")

preds_test[preds_test.view(preds_test.shape[0], -1).sum(-1) < noise_th, ...] = 0.0


# Generate rle encodings (images are first converted to the original size)
preds_test = (preds_test > best_thr).long().numpy()
rles = []
for p in progress_bar(preds_test):
    im = PIL.Image.fromarray((p.T * 255).astype(np.uint8)).resize((1024, 1024))
    im = np.asarray(im)
    rles.append(mask2rle(im, 1024, 1024))


# In[21]:


ids = [o.stem for o in data.test_ds.items]
sub_df = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
sub_df.loc[sub_df.EncodedPixels == "", "EncodedPixels"] = "-1"
sub_df.to_csv(f"Submissions/submission-fastai-{model_name}.csv", index=False)
sub_df.head()


# In[ ]:

