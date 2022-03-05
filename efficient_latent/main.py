import torch
from .models import EfficientNet_B2


def load_model(model_file_path: str = None, device: str = None) -> torch.nn.Module:
    if device is None:
        torch_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)

    # Instantiate model
    model = EfficientNet_B2()
    model = model.to(torch_device)

    if model_file_path is None:
        # Download model
        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/richermans/HEAR2021_EfficientLatent/releases/download/v0.0.1/effb2.pt',
            progress=True)
        model.load_state_dict(state_dict, strict=True)
    else:
        # Set model weights using checkpoint file
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)

    model.sample_rate = 16000  # Input sample rate
    model.scene_embedding_size = 1408
    model.timestamp_embedding_size = 1408
    return model


def get_scene_embeddings(x: torch.Tensor, model: torch.nn.Module, training_mode: bool = False):
    if training_mode:
        embeddings = model.clip_embedding(x)
    else:        
        model.eval()
        with torch.no_grad():
            embeddings = model.clip_embedding(x)
    return embeddings


def get_timestamp_embeddings(x: torch.Tensor, model: torch.nn.Module, keep_timesteps: bool = True, training_mode: bool = False):
    if training_mode:
        time_output, time_stamps = model.segment_embedding(x, keep_timesteps)
    else:
        model.eval()
        with torch.no_grad():
            time_output, time_stamps = model.segment_embedding(x, keep_timesteps)
    return time_output, time_stamps


def get_embeddings(x: torch.Tensor, model: torch.nn.Module, keep_timesteps: bool = True, training_mode: bool = False):
    if training_mode:
        clip_output, time_output, time_stamps = model.get_embeddings(x, keep_timesteps)
    else:
        model.eval()
        with torch.no_grad():
            clip_output, time_output, time_stamps = model.get_embeddings(x, keep_timesteps)
    return clip_output, time_output, time_stamps
