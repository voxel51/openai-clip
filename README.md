# OpenAI CLIP

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/fiftyone-community)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-purple?style=flat&logo=huggingface)](https://huggingface.co/Voxel51)
[![Voxel51 Blog](https://img.shields.io/badge/Voxel51_Blog-ff6d04?style=flat)](https://voxel51.com/blog)
[![Newsletter](https://img.shields.io/badge/Newsletter-BE5B25?logo=mail.ru&logoColor=white)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![LinkedIn](https://img.shields.io/badge/In-white?style=flat&label=Linked&labelColor=blue)](https://www.linkedin.com/company/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-000000?logo=x&logoColor=white)](https://x.com/voxel51)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)

</div>

Wrapper around [OpenAI's CLIP model](https://github.com/openai/CLIP) for the
FiftyOne Model Zoo.

## Example usage

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=50,
    shuffle=True,
)

foz.register_zoo_model_source("https://github.com/voxel51/openai-clip")
model = foz.load_zoo_model(
    "voxel51/clip-vit-base32-torch",
    text_prompt="A photo of a",
    classes=["person", "dog", "cat", "bird", "car", "tree", "chair"],
)

dataset.apply_model(model, label_field="predictions")
```
