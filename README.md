# OpenAI CLIP

Wrapper for [OpenAI's CLIP model](https://github.com/openai/CLIP) for the
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
