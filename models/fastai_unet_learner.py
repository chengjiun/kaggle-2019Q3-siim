# ### Model

# The model used in this kernel is based on U-net like architecture with ResNet34 encoder. To boost the model performance, Hypercolumns are incorporated into DynamicUnet fast.ai class (see code below). The idea of Hypercolumns is schematically illustrated in the following figure. ![](https://i.ibb.co/3y7f8rj/Hypercolumns1.png)
# Each upscaling block is connected to the output layer through linear resize to the original image size. So the final image is produced based on concatenation of U-net output with resized outputs of intermediate layers. These skip-connections provide a shortcut for gradient flow improving model performance and convergence speed. Since intermediate layers have many channels, their upscaling and use as an input for the final layer would introduce a significant overhead in terms the computational time and memory. Therefore, 3x3 convolutions are applied (factorization) before the resize to reduce the number of channels.
# Further details on Hypercolumns can be found [here](http://home.bharathh.info/pubs/pdfs/BharathCVPR2015.pdf) and [here](https://towardsdatascience.com/review-hypercolumn-instance-segmentation-367180495979). Below the fast.ai code modified to incorporate Hypercolumns.

# from fastai.callbacks import SaveModelCallback
import torch
import torch.nn as nn

# fastai vision functions
from fastai.vision import (
    to_device,
    in_channels,
    create_body,
    ifnone,
    apply_init,
    conv2d,
    conv_layer,
    batchnorm_2d,
    PixelShuffle_ICNR,
    MergeLayer,
    res_block,
    SigmoidRange,
    Collection,
    Tensor,
    SequentialEx,
    Optional,
    Tuple,
    DataBunch,
    Callable,
    NormType,
    SplitFuncOrIdxList,
    Union,
    Any,
    OptimWrapper,
    Floats,
    dataclass,
    Learner,
    LearnerCallback,
    Rank0Tensor,
)
import torch.nn.functional as F

# from fastai.vision import *
from fastai.vision.learner import cnn_config
from fastai.callbacks.hooks import (
    model_sizes,
    hook_outputs,
    dummy_eval,
    Hook,
    _hook_inner,
)
from fastai.vision.models.unet import _get_sfs_idxs, UnetBlock, DynamicUnet


class Hcolumns(nn.Module):
    def __init__(self, hooks: Collection[Hook], nc: Collection[int] = None):
        super(Hcolumns, self).__init__()
        self.hooks = hooks
        self.n = len(self.hooks)
        self.factorization = None
        if nc is not None:
            self.factorization = nn.ModuleList()
            for i in range(self.n):
                self.factorization.append(
                    nn.Sequential(
                        conv2d(nc[i], nc[-1], 3, padding=1, bias=True),
                        conv2d(nc[-1], nc[-1], 3, padding=1, bias=True),
                    )
                )
                # self.factorization.append(conv2d(nc[i],nc[-1],3,padding=1,bias=True))

    def forward(self, x: Tensor):
        # n = len(self.hooks)
        out = [
            F.interpolate(
                self.hooks[i].stored
                if self.factorization is None
                else self.factorization[i](self.hooks[i].stored),
                scale_factor=2 ** (self.n - i),
                mode="bilinear",
                align_corners=False,
            )
            for i in range(self.n)
        ] + [x]
        return torch.cat(out, dim=1)


class DynamicUnet_Hcolumns(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        blur: bool = False,
        blur_final=True,
        self_attention: bool = False,
        y_range: Optional[Tuple[float, float]] = None,
        last_cross: bool = True,
        bottle: bool = False,
        sz=256,
        **kwargs,
    ):
        imsize = (sz, sz)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(
            conv_layer(ni, ni * 2, **kwargs), conv_layer(ni * 2, ni, **kwargs)
        ).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        self.hc_hooks = [Hook(layers[-1], _hook_inner, detach=False)]
        hc_c = [x.shape[1]]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            # do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block = UnetBlock(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                **kwargs,
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)
            self.hc_hooks.append(Hook(layers[-1], _hook_inner, detach=False))
            hc_c.append(x.shape[1])

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        hc_c.append(ni)
        layers.append(Hcolumns(self.hc_hooks, hc_c))
        layers += [
            conv_layer(ni * len(hc_c), n_classes, ks=1, use_activ=False, **kwargs)
        ]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()


def unet_learner(
    data: DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType,
    split_on: Optional[SplitFuncOrIdxList] = None,
    blur: bool = False,
    self_attention: bool = False,
    y_range: Optional[Tuple[float, float]] = None,
    last_cross: bool = True,
    bottle: bool = False,
    cut: Union[int, Callable] = None,
    hypercolumns=True,
    **learn_kwargs: Any,
) -> Learner:
    "Build Unet learner from `data` and `arch`."
    meta = cnn_config(arch)
    body = create_body(arch, pretrained, cut)
    M = DynamicUnet_Hcolumns if hypercolumns else DynamicUnet
    model = to_device(
        M(
            body,
            n_classes=data.c,
            blur=blur,
            blur_final=blur_final,
            self_attention=self_attention,
            y_range=y_range,
            norm_type=norm_type,
            last_cross=last_cross,
            bottle=bottle,
        ),
        data.device,
    )
    learn = Learner(data, model, **learn_kwargs)
    learn.split(ifnone(split_on, meta["split"]))
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn


# Accumulation of gradients to overcome the problem of too small batches. The code is mostly based on [this post](https://forums.fast.ai/t/accumulating-gradients/33219/25) with slight adjustment to work with mean reduction.
class AccumulateOptimWrapper(OptimWrapper):
    def step(self):
        pass

    def zero_grad(self):
        pass

    def real_step(self):
        super().step()

    def real_zero_grad(self):
        super().zero_grad()


def acc_create_opt(self, lr: Floats, wd: Floats = 0.0):
    "Create optimizer with `lr` learning rate and `wd` weight decay."
    self.opt = AccumulateOptimWrapper.create(
        self.opt_func,
        lr,
        self.layer_groups,
        wd=wd,
        true_wd=self.true_wd,
        bn_wd=self.bn_wd,
    )


@dataclass
class AccumulateStep(LearnerCallback):
    """
    Does accumlated step every nth step by accumulating gradients
    """

    def __init__(self, learn: Learner, n_step: int = 1):
        super().__init__(learn)
        self.n_step = n_step

    def on_epoch_begin(self, **kwargs):
        "init samples and batches, change optimizer"
        self.acc_batches = 0

    def on_batch_begin(self, last_input, last_target, **kwargs):
        "accumulate samples and batches"
        self.acc_batches += 1

    def on_backward_end(self, **kwargs):
        "step if number of desired batches accumulated, reset samples"
        if (self.acc_batches % self.n_step) == self.n_step - 1:
            for p in self.learn.model.parameters():
                if p.requires_grad:
                    p.grad.div_(self.acc_batches)

            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0

    def on_epoch_end(self, **kwargs):
        "step the rest of the accumulated grads"
        if self.acc_batches > 0:
            for p in self.learn.model.parameters():
                if p.requires_grad:
                    p.grad.div_(self.acc_batches)
            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0


# batch size = 8
def set_BN_momentum(model, momentum=0.1 * 8 / 64):
    for i, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            layer.momentum = momentum


# A slight modification of the default dice metric to make it comparable with the competition metric: dice is computed for each image independently, and dice of empty image with zero prediction is 1. Also I use noise removal and similar threshold as in my prediction pipline.
def dice(
    best_thr0,
    noise_th,
    input: Tensor,
    targs: Tensor,
    iou: bool = False,
    eps: float = 1e-8,
) -> Rank0Tensor:
    n = targs.shape[0]
    input = torch.softmax(input, dim=1)[:, 1, ...].view(n, -1)
    input = (input > best_thr0).long()
    input[input.sum(-1) < noise_th, ...] = 0.0
    # input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n, -1)
    intersect = (input * targs).sum(-1).float()
    union = (input + targs).sum(-1).float()
    if not iou:
        return ((2.0 * intersect + eps) / (union + eps)).mean()
    else:
        return ((intersect + eps) / (union - intersect + eps)).mean()


# dice for threshold selection
def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds + targs).sum(-1).float()
    u0 = union == 0
    intersect[u0] = 1
    union[u0] = 2
    return 2.0 * intersect / union
