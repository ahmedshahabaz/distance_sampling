# hf_midas_patch.py
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
from zoedepth.models.base_models.midas import MidasCore, MIDAS_SETTINGS

_orig_build = MidasCore.build

@staticmethod
def _build_with_hf(
    midas_model_type="DPT_BEiT_L_384",
    train_midas=False,
    use_pretrained_midas=True,
    fetch_features=False,
    freeze_bn=True,
    force_keep_ar=False,
    force_reload=False,
    **kwargs
):
    # parse img_size exactly like original
    if "img_size" in kwargs:
        kwargs = MidasCore.parse_img_size(kwargs)
    img_size = kwargs.pop("img_size", [384, 384])

    # 1) Load DPT from HF
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
    dpt = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")
    dpt = dpt.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    # 2) Create the MidasCore wrapper around it
    #    We store the HF processor on the core so forward() can call it.
    core = MidasCore(
        dpt,
        trainable=train_midas,
        fetch_features=fetch_features,
        freeze_bn=freeze_bn,
        img_size=img_size,
        **kwargs,
    )
    core.processor = processor
    core.set_output_channels(midas_model_type)
    return core

# patch it in
MidasCore.build = _build_with_hf
