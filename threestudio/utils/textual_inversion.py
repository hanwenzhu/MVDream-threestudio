import torch
import torch.nn as nn
from diffusers.loaders import TextualInversionLoaderMixin
from omegaconf import OmegaConf

import threestudio
from threestudio.utils.typing import *


def load_textual_inversion(
    pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    tokenizer: nn.Module,
    text_encoder: nn.Module,
    token: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """Load Textual Inversion embeddings into `tokenizer` and `text_encoder`."""

    threestudio.info(f"Loading textual inversion embeddings from {pretrained_model_name_or_path}")

    # If pretrained_model_name_or_path is an OmegaConf list config type, it is not registered as an instance of `list`
    if OmegaConf.is_config(pretrained_model_name_or_path):
        pretrained_model_name_or_path = OmegaConf.to_container(pretrained_model_name_or_path)
    
    # HACK: calling the `TextualInversionLoaderMixin::load_textual_inversion` method not from a pipeline.
    # This mixin class was intended to be inherited by a DiffusionPipeline class, but here it is used as a bare class.
    # This might fail on future versions of diffusers, and an alternative is to simply copy code from
    # `diffusers/loaders/textual_inversion.py` that loads textual inversion to a tokenizer and a text encoder.
    loader = TextualInversionLoaderMixin()

    # pretend to be a DiffusionPipeline
    loader.tokenizer = tokenizer
    loader.text_encoder = text_encoder
    loader.components = {}

    loader.load_textual_inversion(
        pretrained_model_name_or_path,
        token=token,
        **kwargs
    )
