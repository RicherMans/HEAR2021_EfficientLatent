# Submission to the HEAR 2021 Challenge

For model evaluation, `python=3.7` and `cuda10.2` with `cudnn7.6.5` have been tested.

The work uses a mixed supervised and self-supervised training regime.

## Usage


First install the package:

```
python3 -m pip install git+https://github.com/richermans/HEAR2021_EfficientLatent.git
```


Then just use it:

```python
import torch
import efficient_latent

model = efficient_latent.load_model()

audio = torch.randn(1, 16000) # Sampling rate is 16000

time_embeddings = efficient_latent.get_timestamp_embeddings(audio, model)
clip_embeddings = efficient_latent.get_scene_embeddings(audio, model)
```

