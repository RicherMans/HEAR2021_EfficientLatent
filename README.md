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

clip_embeddings, time_embeddings, time_stamps = efficient_latent.get_embeddings(audio, model)
```


# Results

Notable [results](https://neuralaudio.ai/hear2021-results.html) of this model in the challenge are:


## DCASE2016 Task 2

```
Team Name	Submission	Event Onset FMS	Segment Error Rate
CP-JKU	base2levelmel	0.925	0.099
CP-JKU	base2level	0.913	0.102
MARL + Soundsensing	openl3_hear	0.833	0.174
NTU-GURA	fusion_cat_xwc	0.826	0.145
NTU-GURA	fusion_cat_xwc_time	0.826	0.145
NTU-GURA	fusion_hubert_xlarge	0.826	0.150
NTU-GURA	fusion_wav2vec2	0.798	0.163
*RedRice*	efficient_latent	0.790	0.231
```

## Beijing Opera

```
Team Name	Submission	Accuracy
MARL + Soundsensing	openl3_hear	0.975
NTU-GURA	fusion_cat_xwc	0.966
CP-JKU	base	0.966
CP-JKU	base2level	0.966
CP-JKU	base2levelmel	0.966
NTU-GURA	fusion_cat_xwc_time	0.962
RedRice	efficient_latent	0.953
```

## CREMA-D

```
Team Name	Submission	Accuracy
Logitech AI	serab_byols	0.535
*RedRice*	efficient_latent	0.502
CVSSP	panns_hear	0.440
HEAR	wav2vec2	0.434
```

## ESC-50


```
Team Name	Submission	Accuracy
CP-JKU	base	0.947
CP-JKU	base2level	0.947
CP-JKU	base2levelmel	0.947
*RedRice*	efficient_latent	0.935
CVSSP	panns_hear	0.909
Soundsensing	yamnet_hear	0.838
```

## FSD50k

```
Team Name	Submission	mAP	d'
CP-JKU	base	0.640	2.643
*RedRice*	efficient_latent	0.607	2.538
CP-JKU	base2levelmel	0.558	2.312
CP-JKU	base2level	0.537	2.292
Logitech AI	serab_byols	0.509	2.218
MARL + Soundsensing	openl3_hear	0.447	2.117
```

## GTZAN Genre

```
Team Name	Submission	Accuracy
MARL + Soundsensing	openl3_hear	0.796
RedRice	efficient_latent	0.782
Logitech AI	serab_byols	0.723
CVSSP	panns_hear	0.660
```



