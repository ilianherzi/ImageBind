"""Data loading and transformation utilities for ImageBind multimodal inputs.

This module provides functions to load and preprocess data from different modalities:
- Vision (images) - using torchvision and PIL
- Text - using BPE tokenization
- Audio (waveforms to mel spectrograms) - using torchaudio
- Video (with spatial and temporal sampling) - using OpenCV (cv2)

All pytorchvideo dependencies have been removed and replaced with:
- OpenCV (cv2) for video loading
- Custom implementations for clip sampling, frame subsampling, and resizing
"""

from __future__ import annotations

import io
import logging
import math
from typing import Optional, BinaryIO

import cv2
import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from torchvision import transforms

from imagebind.models.multimodal_preprocessors import SimpleTokenizer

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds


def return_bpe_path() -> str:
    """Return the path to the BPE (Byte Pair Encoding) vocabulary file.

    Returns:
        Path to the BPE vocabulary file for text tokenization.
    """
    return pkg_resources.resource_filename(
        "imagebind", "bpe/bpe_simple_vocab_16e6.txt.gz"
    )


def waveform2melspec(
    waveform: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int,
    target_length: int,
) -> torch.Tensor:
    """Convert audio waveform to mel-frequency spectrogram.

    This function:
    1. Mean-centers the waveform
    2. Computes mel-frequency filter bank features using Kaldi-compatible settings
    3. Pads or crops to target length
    4. Returns as a single-channel image-like tensor

    Based on: https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102

    Args:
        waveform: Audio waveform tensor of shape (channels, samples).
        sample_rate: Audio sampling rate in Hz.
        num_mel_bins: Number of mel-frequency bins.
        target_length: Target number of frames for the output spectrogram.

    Returns:
        Mel spectrogram tensor of shape (1, num_mel_bins, target_length),
        formatted as a single-channel image.

    Warnings:
        Logs a warning if padding/cropping exceeds 20% of the original length.
    """
    # Mean-center the waveform
    waveform -= waveform.mean()

    # Compute mel-frequency filter bank features
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )

    # Convert from (num_frames, mel_bins) to (mel_bins, num_frames)
    fbank = fbank.transpose(0, 1)

    # Pad or crop to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames

    # Warn if padding/cropping is more than 20% of original length
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )

    # Apply padding or cropping
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]

    # Add channel dimension: (mel_bins, num_frames) -> (1, mel_bins, num_frames)
    fbank = fbank.unsqueeze(0)
    return fbank


def sample_clip_timepoints(
    duration: float,
    clip_duration: float,
    clips_per_video: int,
) -> list[tuple[float, float]]:
    """Sample clip timepoints uniformly from audio/video duration.

    This is a replacement for pytorchvideo's ConstantClipsPerVideoSampler.
    It divides the duration into evenly-spaced clips.

    Strategy:
    - If clips_per_video == 1: Takes center clip
    - If clips_per_video > 1: Evenly spaces clips across duration
    - Clips may overlap if clips_per_video * clip_duration > duration

    Args:
        duration: Total duration in seconds.
        clip_duration: Duration of each clip in seconds.
        clips_per_video: Number of clips to extract.

    Returns:
        List of (start_time, end_time) tuples in seconds for each clip.

    Examples:
        >>> sample_clip_timepoints(10.0, 2.0, 3)
        [(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]

        >>> sample_clip_timepoints(10.0, 2.0, 1)
        [(4.0, 6.0)]  # Center clip
    """
    if clips_per_video == 1:
        # Single clip: take from center
        start_time = max(0, (duration - clip_duration) / 2)
        return [(start_time, min(start_time + clip_duration, duration))]

    # Multiple clips: evenly space them
    # Calculate spacing between clip start times
    max_start = max(0, duration - clip_duration)
    interval = max_start / max(1, clips_per_video - 1)

    clips = []
    for i in range(clips_per_video):
        start_time = min(i * interval, max_start)
        end_time = min(start_time + clip_duration, duration)
        clips.append((start_time, end_time))

    return clips


def load_and_transform_vision_data(
    image_paths: Optional[list[str]],
    device: torch.device | str,
    image_size: int = 224,
) -> Optional[torch.Tensor]:
    """Load and preprocess images for vision encoder.

    Applies standard ImageNet normalization and resizing.

    Args:
        image_paths: List of file paths to images, or None.
        device: Device to place the tensors on (e.g., 'cuda' or 'cpu').
        image_size: Target size for image resize and crop.

    Returns:
        Tensor of shape (batch_size, 3, image_size, image_size) containing preprocessed images,
        or None if image_paths is None.
    """
    if image_paths is None:
        return None

    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)

    return torch.stack(image_outputs, dim=0)


def load_and_transform_text(
    text: Optional[list[str]],
    device: torch.device | str,
) -> Optional[torch.Tensor]:
    """Tokenize and encode text for text encoder.

    Uses BPE (Byte Pair Encoding) tokenization.

    Args:
        text: List of text strings to tokenize, or None.
        device: Device to place the tensors on (e.g., 'cuda' or 'cpu').

    Returns:
        Tensor of shape (batch_size, context_length) containing token indices,
        or None if text is None.
    """
    if text is None:
        return None

    tokenizer = SimpleTokenizer(bpe_path=return_bpe_path())
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_audio_data(
    audio_paths: Optional[list[str]],
    device: torch.device | str,
    num_mel_bins: int = 128,
    target_length: int = 204,
    sample_rate: int = 16000,
    clip_duration: float = 2.0,
    clips_per_video: int = 3,
    mean: float = -4.268,
    std: float = 9.138,
) -> Optional[torch.Tensor]:
    """Load and preprocess audio files to mel spectrograms.

    This function:
    1. Loads audio waveforms and resamples to target sample rate
    2. Splits audio into multiple clips uniformly across duration
    3. Converts each clip to mel spectrogram
    4. Normalizes spectrograms using provided mean and std

    Args:
        audio_paths: List of file paths to audio files, or None.
        device: Device to place the tensors on (e.g., 'cuda' or 'cpu').
        num_mel_bins: Number of mel-frequency bins for spectrogram.
        target_length: Target number of frames for each spectrogram.
        sample_rate: Target sampling rate in Hz for resampling.
        clip_duration: Duration of each clip in seconds.
        clips_per_video: Number of clips to sample from each audio file.
        mean: Mean for normalization of mel spectrograms.
        std: Standard deviation for normalization of mel spectrograms.

    Returns:
        Tensor of shape (batch_size, clips_per_video, 1, num_mel_bins, target_length)
        containing preprocessed audio spectrograms, or None if audio_paths is None.
    """
    if audio_paths is None:
        return None

    audio_outputs = []

    for audio_path in audio_paths:
        # Load audio waveform
        waveform, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )

        # Get clip timepoints
        duration = waveform.size(1) / sample_rate

        all_clips_timepoints = sample_clip_timepoints(
            duration, clip_duration, clips_per_video
        )

        # Extract and process each clip
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        # Normalize spectrograms
        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def load_and_transform_audio_from_waveform(
    waveform: torch.Tensor,
    original_sr: int,
    device: torch.device | str,
    num_mel_bins: int = 128,
    target_length: int = 204,
    sample_rate: int = 16000,
    clip_duration: float = 2.0,
    clips_per_video: int = 3,
    mean: float = -4.268,
    std: float = 9.138,
) -> torch.Tensor:
    """Process audio waveform tensor to mel spectrograms.

    This is useful when you've already loaded audio into memory (e.g., from video
    container or streaming source) and don't want to save to disk first.

    Args:
        waveform: Audio waveform tensor of shape (channels, samples).
        original_sr: Original sampling rate of the waveform.
        device: Device to place the tensors on (e.g., 'cuda' or 'cpu').
        num_mel_bins: Number of mel-frequency bins for spectrogram.
        target_length: Target number of frames for each spectrogram.
        sample_rate: Target sampling rate in Hz for resampling.
        clip_duration: Duration of each clip in seconds.
        clips_per_video: Number of clips to sample from the audio.
        mean: Mean for normalization of mel spectrograms.
        std: Standard deviation for normalization of mel spectrograms.

    Returns:
        Tensor of shape (clips_per_video, 1, num_mel_bins, target_length)
        containing preprocessed audio spectrograms.

    Example:
        >>> # Extract audio from video with ffmpeg or similar
        >>> import torchaudio
        >>> waveform, sr = torchaudio.load("audio.wav")
        >>> spectrograms = load_and_transform_audio_from_waveform(
        ...     waveform, sr, device="cuda"
        ... )
    """
    # Resample if necessary
    if sample_rate != original_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=original_sr, new_freq=sample_rate
        )

    # Get clip timepoints
    duration = waveform.size(1) / sample_rate
    all_clips_timepoints = sample_clip_timepoints(
        duration, clip_duration, clips_per_video
    )

    # Extract and process each clip
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        waveform_melspec = waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    # Normalize spectrograms
    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac).to(device) for ac in all_clips]

    return torch.stack(all_clips, dim=0)


def load_audio_from_bytes(
    audio_bytes: bytes | BinaryIO,
    audio_format: Optional[str] = None,
) -> tuple[torch.Tensor, int]:
    """Load audio from binary data or file-like object.

    This is useful when you have audio data in memory (e.g., extracted from
    an MP4 video container) and want to avoid writing to disk.

    Args:
        audio_bytes: Raw audio bytes or file-like object (e.g., io.BytesIO).
        audio_format: Audio format hint (e.g., "mp3", "wav", "mp4"). If None,
            torchaudio will try to infer it.

    Returns:
        Tuple of (waveform, sample_rate) where waveform has shape (channels, samples).

    Example:
        >>> # From bytes
        >>> with open("audio.mp3", "rb") as f:
        ...     audio_bytes = f.read()
        >>> waveform, sr = load_audio_from_bytes(audio_bytes, audio_format="mp3")

        >>> # From BytesIO (e.g., audio extracted from video)
        >>> import io
        >>> audio_stream = io.BytesIO(extracted_audio_bytes)
        >>> waveform, sr = load_audio_from_bytes(audio_stream)

        >>> # Then process with existing function
        >>> spectrograms = load_and_transform_audio_from_waveform(
        ...     waveform, sr, device="cuda"
        ... )
    """
    # Convert bytes to BytesIO if needed
    if isinstance(audio_bytes, bytes):
        audio_bytes = io.BytesIO(audio_bytes)

    # Load using torchaudio
    # Note: torchaudio.load can accept file-like objects
    waveform, sample_rate = torchaudio.load(audio_bytes, format=audio_format)

    return waveform, sample_rate


def crop_boxes(boxes: torch.Tensor, x_offset: int, y_offset: int) -> torch.Tensor:
    """Adjust bounding box coordinates after spatial cropping.

    Args:
        boxes: Bounding boxes of shape (num_boxes, 4) in format [x1, y1, x2, y2].
        x_offset: Horizontal offset of the crop in pixels.
        y_offset: Vertical offset of the crop in pixels.

    Returns:
        Adjusted bounding boxes of shape (num_boxes, 4) with coordinates
        relative to the cropped region.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(
    images: torch.Tensor,
    size: int,
    spatial_idx: int,
    boxes: Optional[torch.Tensor] = None,
    scale_size: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Perform uniform spatial cropping on images.

    This function supports three crop positions:
    - If width > height: left (0), center (1), or right (2)
    - If height > width: top (0), center (1), or bottom (2)

    Args:
        images: Images tensor of shape (num_frames, channels, height, width)
            or (channels, height, width).
        size: Target size for both height and width of the crop.
        spatial_idx: Crop position index (0=left/top, 1=center, 2=right/bottom).
        boxes: Optional bounding boxes of shape (num_boxes, 4) to crop accordingly.
        scale_size: Optional size to resize images to before cropping.

    Returns:
        Tuple of:
        - Cropped images of shape (num_frames, channels, size, size) or
          (channels, size, size).
        - Cropped bounding boxes of shape (num_boxes, 4) or None.

    Raises:
        AssertionError: If spatial_idx is not in [0, 1, 2].
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)

    # Add batch dimension if needed
    if ndim == 3:
        images = images.unsqueeze(0)

    height = images.shape[2]
    width = images.shape[3]

    # Optional resize before cropping
    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    # Calculate default center crop offsets
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    # Adjust offsets based on spatial_idx
    if height > width:
        # Vertical crops (top, center, bottom)
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        # Horizontal crops (left, center, right)
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size

    # Perform the crop
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None

    # Remove batch dimension if it was added
    if ndim == 3:
        cropped = cropped.squeeze(0)

    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """Apply multiple spatial crops to video frames.

    This module creates multiple views of each video by cropping at different
    spatial positions (left/top, center, right/bottom). This is commonly used
    for data augmentation and test-time augmentation.

    Attributes:
        crop_size: Size of the square crop.
        crops_to_ext: List of spatial indices to extract.
        flipped_crops_to_ext: List of spatial indices to extract from flipped video.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3) -> None:
        """Initialize the SpatialCrop module.

        Args:
            crop_size: Size of the square crop in pixels.
            num_crops: Number of crops to extract (1 for center only, 3 for
                left/center/right or top/center/bottom).

        Raises:
            NotImplementedError: If num_crops is not 1 or 3.
        """
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]  # Center crop only
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply spatial crops to a list of videos.

        Args:
            videos: List of video tensors, each of shape (T, C, H, W).

        Returns:
            List of cropped videos, each of shape (T, C, crop_size, crop_size).
            Length is num_crops times the input length.

        Raises:
            AssertionError: If input is not a list or videos don't have 4 dimensions.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (T,C,H,W)"

        res = []
        for video in videos:
            # Extract crops at specified positions
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])

            # Extract crops from horizontally flipped video if specified
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])

        return res


def subsample_frames_uniformly(video: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Uniformly subsample frames from a video tensor.

    Replacement for pytorchvideo's UniformTemporalSubsample.

    Args:
        video: Video tensor of shape (T, C, H, W).
        num_samples: Number of frames to sample.

    Returns:
        Subsampled video tensor with num_samples frames in shape (T, C, H, W).

    Example:
        >>> video = torch.randn(100, 3, 224, 224)  # 100 frames, 3 channels
        >>> sampled = subsample_frames_uniformly(video, 10)  # 10 frames
        >>> sampled.shape
        torch.Size([10, 3, 224, 224])
    """
    if video.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {video.shape}")

    # Assume (T, C, H, W) format
    num_frames = video.shape[0]
    indices = torch.linspace(0, num_frames - 1, num_samples).long()
    return video[indices, :, :, :]


def resize_short_side_video(
    video: torch.Tensor,
    target_size: int = 224,
) -> torch.Tensor:
    """Resize video so the shorter spatial dimension equals target size.

    Replacement for pytorchvideo's ShortSideScale. Maintains aspect ratio.

    Args:
        video: Video tensor of shape (T, C, H, W).
        target_size: Target size for the shorter spatial dimension.

    Returns:
        Resized video tensor of shape (T, C, new_H, new_W).

    Example:
        >>> video = torch.randn(10, 3, 480, 640)  # 10 frames, 3 channels, 480x640
        >>> resized = resize_short_side_video(video, 224)
        >>> resized.shape  # Shorter side (480) becomes 224
        torch.Size([10, 3, 224, 298])
    """
    T, C, H, W = video.shape

    # Calculate new dimensions maintaining aspect ratio
    if H < W:
        new_h = target_size
        new_w = int(W * target_size / H)
    else:
        new_h = int(H * target_size / W)
        new_w = target_size

    # Reshape to (T*C, 1, H, W) for interpolate (treats each frame-channel independently)
    video_reshaped = video.reshape(T * C, 1, H, W)

    # Resize using bilinear interpolation
    video_resized = torch.nn.functional.interpolate(
        video_reshaped, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Reshape back to (T, C, new_H, new_W)
    return video_resized.reshape(T, C, new_h, new_w)


class NormalizeVideo(nn.Module):
    """Normalize video tensors with mean and standard deviation.

    This is a wrapper around tensor normalization that's compatible
    with video data format (T, C, H, W).
    """

    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        """Initialize the NormalizeVideo module.

        Args:
            mean: Mean values for (R, G, B) channels.
            std: Standard deviation values for (R, G, B) channels.
        """
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize a video tensor.

        Args:
            video: Video tensor of shape (T, C, H, W).

        Returns:
            Normalized video tensor.
        """
        # Move mean/std to same device as video
        if self.mean.device != video.device:
            self.mean = self.mean.to(video.device)
            self.std = self.std.to(video.device)

        return (video - self.mean) / self.std


def load_video_clip(
    video_path: str,
    start_time: float,
    end_time: float,
) -> tuple[torch.Tensor, float]:
    """Load video clip using OpenCV.

    Args:
        video_path: Path to video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Tuple of (video_tensor, fps) where video_tensor has shape (T, H, W, C)
        and values in [0, 255].

    Raises:
        ValueError: If video cannot be opened.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert time to frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Ensure valid range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    # Set position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read frames
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        # If no frames were read, return a single black frame
        raise ValueError("No frames returned")
    # Convert to tensor (T, H, W, C)
    video_tensor = torch.from_numpy(np.stack(frames)).float()

    return video_tensor, fps


def load_and_transform_video_data(
    video_paths: Optional[list[str]],
    device: torch.device | str,
    clip_duration: int = 2,
    clips_per_video: int = 5,
    image_size: int = 224,
    num_crops: int = 1,
    _sample_rate: int = 16000,
) -> Optional[torch.Tensor]:
    """Load and preprocess video files.

    This function performs the following steps:
    1. Loads video using OpenCV (cv2)
    2. Samples multiple clips uniformly across video duration
    3. Uniformly subsamples frames from each clip
    4. Resizes shortest side to image_size while maintaining aspect ratio
    5. Normalizes with ImageNet statistics
    6. Applies spatial crops (left/center/right or top/center/bottom)

    Args:
        video_paths: List of file paths to video files, or None.
        device: Device to place the tensors on (e.g., 'cuda' or 'cpu').
        clip_duration: Duration of each clip in seconds (also used as number of frames).
        clips_per_video: Number of clips to sample from each video.
        image_size: Target size for image resize and spatial crop.
        num_crops: Number of spatial crops to extract (1 for center only, 3 for left/center/right).
        _sample_rate: Unused parameter (kept for API compatibility).

    Returns:
        Tensor of shape (batch_size, clips_per_video * num_crops, 3, clip_duration, image_size, image_size)
        containing preprocessed video clips with spatial crops, or None if
        video_paths is None.

    Raises:
        ValueError: If video cannot be loaded.
    """
    if video_paths is None:
        return None

    video_outputs = []

    # Create normalization transform
    normalizer = NormalizeVideo(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    for video_path in video_paths:
        # Load video to get duration
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        # Sample clip timepoints uniformly
        all_clips_timepoints = sample_clip_timepoints(
            duration, clip_duration, clips_per_video
        )

        all_video = []
        for start_time, end_time in all_clips_timepoints:
            # Load video clip
            video_clip, _ = load_video_clip(video_path, start_time, end_time)

            # video_clip shape: (T, H, W, C) with values in [0, 255]
            # Convert to (T, C, H, W) and normalize to [0, 1]
            video_clip = video_clip.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
            video_clip = video_clip / 255.0

            # Subsample frames uniformly
            video_clip = subsample_frames_uniformly(video_clip, clip_duration)

            # Resize short side to target size
            video_clip = resize_short_side_video(video_clip, image_size)

            # Normalize
            video_clip = normalizer(video_clip)

            all_video.append(video_clip)

        # Apply spatial crops
        all_video = SpatialCrop(image_size, num_crops=num_crops)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)
