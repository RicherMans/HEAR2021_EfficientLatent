import torch
import torch.nn as nn
import torchaudio.transforms as audio_transforms
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils as efficientnet_utils
from typing import Tuple
from einops import rearrange, reduce, repeat


def upsample(x, ratio, target_length):
    x = rearrange(x, 'b t d -> b d t')
    x = repeat(x, '... t -> ... (t r)', r=ratio)
    left_over = target_length - x.shape[-1]
    # Pad leftovoer with reflection, might be only some frames
    if left_over > 0:
        x = torch.nn.functional.pad(x, (0, left_over), 'replicate')
    elif left_over < 0:
        time_len = x.shape[-1]
        startcrop = time_len // 2 - (target_length // 2)
        end_crop = startcrop + target_length
        x = x[..., startcrop:end_crop]
    x = rearrange(x, 'b d t -> b t d')
    return x


# Might use it for other purposes
def overlapping_windows(
        x: torch.Tensor,  # raw wave tensor, ( Batch, Time)
        win_size: int,  # In samples
        hop_size: int,  # In samples
        center: bool = True) -> torch.Tensor:
    if center:
        x = torch.nn.functional.pad(x, (win_size // 2, win_size // 2 - 1),
                                    mode='constant')
    x = rearrange(x, 'batch time -> batch 1 1 time')

    x = torch.nn.functional.unfold(x,
                                   kernel_size=(1, win_size),
                                   stride=(1, hop_size))
    return rearrange(x, 'b t chunks -> chunks b t')


class _EffiNet(EfficientNet):
    """A proxy for efficient net models"""
    def __init__(self,
                 blocks_args=None,
                 global_params=None,
                 embed_dim: int = 1408,
                 **kwargs) -> None:
        super().__init__(blocks_args=blocks_args, global_params=global_params)
        self.n_mels: int = kwargs.get('n_mels', 64)
        self.hop_size: int = kwargs.get('hop_size', 160)
        self.win_size: int = kwargs.get('win_size', 512)
        self.f_min: int = kwargs.get('f_min', 0)
        self.sample_rate = 16_000
        self.hop_size_in_ms = int(self.hop_size / self.sample_rate * 1_000)

        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=self.f_min,
                                            sample_rate=self.sample_rate,
                                            win_length=self.win_size,
                                            n_fft=self.win_size,
                                            hop_length=self.hop_size,
                                            n_mels=self.n_mels),
            audio_transforms.AmplitudeToDB(top_db=120),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x = self.front_end(x)
        input_num_frames = x.shape[-1]  # For later upsampling
        x = rearrange(x, 'b f t -> b 1 f t')
        x = super().extract_features(x)
        return reduce(x, 'b c f t -> b t c', 'mean'), input_num_frames

    def segment_embedding(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_sounds = x.shape[0]
        # Can also process in parallel but that might blow up some memory for a very long clip
        x, num_input_frames = self.forward(x)
        x = upsample(x, ratio=32, target_length=num_input_frames)

        time_steps = torch.arange(0,
            num_input_frames * self.hop_size_in_ms,
            self.hop_size_in_ms)
        # Repeat for each batch/nsounds
        time_steps = repeat(time_steps, 't -> b t', b=n_sounds)
        return x, time_steps

    def clip_embedding(self, x: torch.Tensor) -> torch.Tensor:
        segments, _ = self.forward(x)
        return reduce(segments, 'b t d -> b d', 'mean')


def EfficientNet_B2(**kwargs) -> _EffiNet:
    blocks_args, global_params = efficientnet_utils.get_model_params(
        'efficientnet-b2', {'include_top': False})
    model = _EffiNet(blocks_args=blocks_args,
                     global_params=global_params,
                     embed_dim=1_408,
                     **kwargs)
    model._change_in_channels(1)
    return model


if __name__ == "__main__":
    mdl = EfficientNet_B2()
    y, t = mdl.segment_embedding(torch.randn(1, 16_000))
    print(y.shape, t.shape)
    print(t)
