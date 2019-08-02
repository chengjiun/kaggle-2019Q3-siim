import fastai
from fastai.vision import *
from fastai.vision import open_mask, is_listy, get_transforms
from pathlib import Path
from sklearn.model_selection import KFold


# Setting div=True in open_mask
class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn):
        return open_mask(fn, div=True)


class SegmentationItemList(SegmentationItemList):
    _label_cls = SegmentationLabelList


# Setting transformations on masks to False on test set
def new_transform(
    self, tfms: Optional[Tuple[TfmList, TfmList]] = (None, None), **kwargs
):
    if not tfms:
        tfms = (None, None)
    assert is_listy(tfms) and len(tfms) == 2
    self.train.transform(tfms[0], **kwargs)
    self.valid.transform(tfms[1], **kwargs)
    kwargs["tfm_y"] = False  # Test data has no labels
    if self.test:
        self.test.transform(tfms[1], **kwargs)
    return self


fastai.data_block.ItemLists.transform = new_transform


def get_data(TRAIN, TEST, stats, sz, bs, fold):
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    valid_idx = list(kf.split(list(range(len(Path(TRAIN).ls())))))[fold][1]
    # Create databunch
    data = (
        SegmentationItemList.from_folder(TRAIN)
        .split_by_idx(valid_idx)
        .label_from_func(lambda x: str(x).replace("train", "masks"), classes=[0, 1])
        .add_test(Path(TEST).ls(), label=None)
        .transform(get_transforms(), size=sz, tfm_y=True)
        .databunch(path=Path("."), bs=bs)
        .normalize(stats)
    )
    return data
