"""
OpenAI CLIP wrapper for the FiftyOne Model Zoo.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from packaging.version import Version
import warnings

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

import torch

from .tokenizer import SimpleTokenizer
from .model import build_model


class TorchCLIPModelConfig(fout.TorchImageModelConfig):
    """Configuration for running a :class:`TorchCLIPModel`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        tokenizer_path: the path to the model's tokenizer on disk
        context_length: the model's context length
        text_prompt: the text prompt to use, e.g., ``"A photo of"``
        classes: the list of classes to use for zero-shot prediction
    """

    def __init__(self, d):
        d = self.init(d)
        super().__init__(d)

        self.tokenizer_path = self.parse_string(d, "tokenizer_path")
        self.context_length = self.parse_int(d, "context_length")
        self.text_prompt = self.parse_string(d, "text_prompt")


class TorchCLIPModel(fout.TorchImageModel, fom.PromptMixin):
    """Wrapper for CLIP from https://github.com/openai/CLIP.

    Args:
        config: a :class:`TorchCLIPModelConfig`
    """

    def __init__(self, config):
        super().__init__(config)

        self._tokenizer = SimpleTokenizer(config.tokenizer_path)
        self._text_features = None

    @property
    def can_embed_prompts(self):
        return True

    def embed_prompt(self, prompt):
        return self.embed_prompts([prompt])[0]

    def embed_prompts(self, prompts):
        return self._embed_prompts(prompts).detach().cpu().numpy()

    def _load_model(self, config):
        with open(config.model_path, "rb") as f:
            model = torch.jit.load(f, map_location=self.device).eval()

        return build_model(model.state_dict()).to(self.device).float()

    def _embed_prompts(self, prompts):
        # source: https://github.com/openai/CLIP/blob/main/clip/clip.py
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self._tokenizer.encode(p) + [eot_token]
            for p in prompts
        ]

        if Version(torch.__version__) < Version("1.8.0"):
            dtype = torch.long
        else:
            dtype = torch.int

        text_features = torch.zeros(
            len(all_tokens),
            self.config.context_length,
            dtype=dtype,
            device=self.device,
        )

        for i, (prompt, tokens) in enumerate(zip(prompts, all_tokens)):
            if len(tokens) > self.config.context_length:
                tokens = tokens[: self.config.context_length]
                tokens[-1] = eot_token
                msg = (
                    "Truncating prompt '%s'; too long for context length '%d'"
                    % (prompt, self.config.context_length)
                )
                warnings.warn(msg)

            text_features[i, : len(tokens)] = torch.tensor(tokens)

        with torch.no_grad():
            return self._model.encode_text(text_features)

    def _get_text_features(self):
        if self._text_features is None:
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            self._text_features = self._embed_prompts(prompts)

        return self._text_features

    def _get_class_logits(self, text_features, image_features):
        # source: https://github.com/openai/CLIP/blob/main/README.md
        image_features = image_features / image_features.norm(
            dim=1, keepdim=True
        )
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self._model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        if self._preprocess:
            imgs = [self._transforms(img) for img in imgs]

        if isinstance(imgs, (list, tuple)):
            imgs = torch.stack(imgs)

        height, width = imgs.size()[-2:]
        frame_size = (width, height)

        if self._using_gpu:
            imgs = imgs.cuda()

        text_features = self._get_text_features()
        image_features = self._model.encode_image(imgs)
        output, _ = self._get_class_logits(text_features, image_features)

        if self.has_logits:
            self._output_processor.store_logits = self.store_logits

        return self._output_processor(
            output, frame_size, confidence_thresh=self.config.confidence_thresh
        )
