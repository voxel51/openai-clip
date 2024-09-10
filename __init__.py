"""
OpenAI CLIP model from https://github.com/openai/CLIP.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os

import eta.core.web as etaw

from .zoo import TorchCLIPModelConfig, TorchCLIPModel


logger = logging.getLogger(__name__)


MODEL_URL = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
TOKENIZER_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CLASSES = "aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor"


def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    if model_name != "voxel51/clip-vit-base32-torch":
        raise ValueError("Unsupported model name '%s'" % model_name)

    logger.info("Downloading model...")
    etaw.download_file(MODEL_URL, path=model_path)

    logger.info("Downloading tokenizer...")
    etaw.download_file(TOKENIZER_URL, path=_get_tokenizer_path(model_path))


def load_model(model_name, model_path, text_prompt="A photo of", classes=None):
    """Loads the model.

    Args:
        model_name: the name of the model to load, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which the model was
            donwloaded, as declared by the ``base_filename`` field of the
            manifest
        text_prompt ("A photo of"): the text prompt to use
        classes (None): the list of classes to use for zero-shot prediction.
            By default, the VOC classes are used

    Returns:
        a :class:`fiftyone.core.models.Model`
    """
    if model_name != "voxel51/clip-vit-base32-torch":
        raise ValueError("Unsupported model name '%s'" % model_name)

    if classes is None:
        classes = DEFAULT_CLASSES.split(",")

    config = TorchCLIPModelConfig(
        dict(
            model_path=model_path,
            tokenizer_path=_get_tokenizer_path(model_path),
            context_length=77,
            text_prompt=text_prompt,
            classes=classes,
            output_processor_cls="fiftyone.utils.torch.ClassifierOutputProcessor",
            image_size=[224, 224],
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            embeddings_layer="visual",
        )
    )
    return TorchCLIPModel(config)


def _get_tokenizer_path(model_path):
    model_dir = os.path.dirname(model_path)
    return os.path.join(model_dir, "clip_bpe_simple_vocab_16e6.txt.gz")
