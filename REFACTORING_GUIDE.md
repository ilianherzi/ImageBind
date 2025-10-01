# ImageBind Data Loading Refactoring Guide

## Overview

The `imagebind/data.py` file contains data loading and preprocessing utilities for ImageBind's multimodal inputs. This guide explains what each component does and how to refactor away from `pytorchvideo` dependencies.

## Current Dependencies on pytorchvideo

### 1. **Video Loading: `EncodedVideo`**
- **What it does**: Loads and decodes video files using the decord backend
- **Used in**: `load_and_transform_video_data()`
- **Refactoring options**:
  - Use `torchvision.io.read_video()` - simpler but less flexible
  - Use `decord` directly - more control over decoding
  - Use `opencv-python` (cv2) - widely compatible

### 2. **Clip Sampling: `ConstantClipsPerVideoSampler`**
- **What it does**: Samples a fixed number of clips from a video at regular intervals
- **Logic**: Divides video duration by `clips_per_video` to get clip spacing
- **Refactoring**: Implement manual clip sampling:
  ```python
  def sample_clips(duration, clip_duration, clips_per_video):
      interval = (duration - clip_duration) / max(1, clips_per_video - 1)
      return [(i * interval, i * interval + clip_duration) 
              for i in range(clips_per_video)]
  ```

### 3. **Frame Sampling: `UniformTemporalSubsample`**
- **What it does**: Uniformly samples `num_samples` frames from a clip
- **Refactoring**: Use simple indexing:
  ```python
  num_frames = video.shape[1]  # Assuming (C, T, H, W)
  indices = torch.linspace(0, num_frames - 1, num_samples).long()
  sampled_video = video[:, indices, :, :]
  ```

### 4. **Video Resizing: `ShortSideScale`**
- **What it does**: Resizes video so the shorter spatial dimension equals target size
- **Refactoring**: Use torchvision.transforms:
  ```python
  from torchvision.transforms import Resize
  
  def resize_short_side(video, target_size=224):
      # video shape: (C, T, H, W)
      h, w = video.shape[-2:]
      if h < w:
          new_h, new_w = target_size, int(w * target_size / h)
      else:
          new_h, new_w = int(h * target_size / w), target_size
      
      # Resize each frame
      return torch.nn.functional.interpolate(
          video.permute(1, 0, 2, 3),  # (T, C, H, W) for interpolate
          size=(new_h, new_w),
          mode='bilinear',
          align_corners=False
      ).permute(1, 0, 2, 3)  # Back to (C, T, H, W)
  ```

## File Structure Breakdown

### Data Loading Functions

#### 1. `load_and_transform_vision_data()`
**Purpose**: Load and preprocess static images

**Pipeline**:
1. Resize to 224x224 (bicubic interpolation)
2. Center crop
3. Convert to tensor
4. Normalize with ImageNet stats

**No refactoring needed** - uses only torchvision.transforms

---

#### 2. `load_and_transform_text()`
**Purpose**: Tokenize text strings

**Pipeline**:
1. Load BPE tokenizer
2. Tokenize each text string
3. Return token indices

**No refactoring needed** - uses custom SimpleTokenizer

---

#### 3. `load_and_transform_audio_data()`
**Purpose**: Load audio and convert to mel spectrograms

**Pipeline**:
1. Load audio with torchaudio
2. Resample to 16kHz if needed
3. Split into clips using `ConstantClipsPerVideoSampler` ⚠️
4. Convert each clip to mel spectrogram (128 bins, 204 frames)
5. Normalize

**Refactoring needed**:
- Replace `ConstantClipsPerVideoSampler` with manual clip extraction
- Everything else uses standard libraries (torchaudio)

---

#### 4. `load_and_transform_video_data()`
**Purpose**: Load and preprocess videos with multiple spatial views

**Pipeline**:
1. Load video with `EncodedVideo` ⚠️
2. Sample clips with `ConstantClipsPerVideoSampler` ⚠️
3. Uniformly subsample frames with `UniformTemporalSubsample` ⚠️
4. Resize with `ShortSideScale` ⚠️
5. Normalize with ImageNet stats
6. Create 3 spatial crops (left, center, right OR top, center, bottom)

**Heavy refactoring needed** - see replacement implementation below

---

### Helper Functions

#### `waveform2melspec()`
Converts audio waveform to mel spectrogram using Kaldi-compatible settings.
- **No refactoring needed** - uses torchaudio

#### `get_clip_timepoints()`
Extracts clip boundaries from a video.
- **Needs refactoring** - depends on `ConstantClipsPerVideoSampler`

#### `crop_boxes()` & `uniform_crop()`
Spatial cropping utilities for videos/images.
- **No refactoring needed** - pure PyTorch operations

#### `SpatialCrop` class
Applies multiple spatial crops (left/center/right) to videos.
- **No refactoring needed** - pure PyTorch operations

#### `NormalizeVideo` class
Normalizes video tensors with mean/std.
- **No refactoring needed** - wraps torchvision.transforms.Normalize

---

## Recommended Refactoring Approach

### Step 1: Replace Video Loading

Replace `EncodedVideo` with direct decord usage or torchvision:

```python
import decord
from decord import VideoReader, cpu

def load_video_decord(video_path):
    """Load video using decord directly."""
    vr = VideoReader(video_path, ctx=cpu(0))
    return vr  # Can index into this: vr[frame_idx]
```

Or use torchvision:

```python
from torchvision.io import read_video

def load_video_torchvision(video_path):
    """Load video using torchvision."""
    video, audio, info = read_video(video_path, pts_unit='sec')
    # video shape: (T, H, W, C) - need to permute to (C, T, H, W)
    return video.permute(3, 0, 1, 2) / 255.0
```

### Step 2: Implement Custom Clip Sampling

```python
def sample_clips_uniform(duration, clip_duration, clips_per_video):
    """Sample clips uniformly across video duration."""
    if clips_per_video == 1:
        # Center clip
        start = (duration - clip_duration) / 2
        return [(start, start + clip_duration)]
    
    # Multiple clips evenly spaced
    interval = (duration - clip_duration) / (clips_per_video - 1)
    return [(i * interval, i * interval + clip_duration) 
            for i in range(clips_per_video)]
```

### Step 3: Implement Frame Subsampling

```python
def subsample_frames(video_tensor, num_samples):
    """Uniformly subsample frames from video.
    
    Args:
        video_tensor: (C, T, H, W) tensor
        num_samples: Number of frames to sample
    
    Returns:
        Subsampled video: (C, num_samples, H, W)
    """
    num_frames = video_tensor.shape[1]
    indices = torch.linspace(0, num_frames - 1, num_samples).long()
    return video_tensor[:, indices, :, :]
```

### Step 4: Implement Short-Side Scaling

```python
def resize_short_side_video(video, target_size=224):
    """Resize video so shortest side matches target_size.
    
    Args:
        video: (C, T, H, W) tensor
        target_size: Target size for short side
    
    Returns:
        Resized video: (C, T, new_H, new_W)
    """
    C, T, H, W = video.shape
    
    # Calculate new dimensions
    if H < W:
        new_h = target_size
        new_w = int(W * target_size / H)
    else:
        new_h = int(H * target_size / W)
        new_w = target_size
    
    # Reshape for interpolate: (C*T, 1, H, W) - treat as batch of images
    video_reshaped = video.reshape(C * T, 1, H, W)
    
    # Resize
    video_resized = torch.nn.functional.interpolate(
        video_reshaped,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Reshape back: (C, T, new_H, new_W)
    return video_resized.reshape(C, T, new_h, new_w)
```

### Step 5: Put It All Together

```python
def load_and_transform_video_data_refactored(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
):
    """Load and preprocess videos without pytorchvideo."""
    if video_paths is None:
        return None
    
    video_outputs = []
    
    for video_path in video_paths:
        # 1. Load video (using decord or torchvision)
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        duration = len(vr) / fps
        
        # 2. Sample clips
        clip_timepoints = sample_clips_uniform(
            duration, clip_duration, clips_per_video
        )
        
        all_video = []
        for start, end in clip_timepoints:
            # Get frame indices for this clip
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            
            # Load frames
            frame_indices = list(range(start_frame, end_frame))
            frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
            
            # Convert to tensor and normalize
            video_clip = torch.from_numpy(frames).float() / 255.0
            video_clip = video_clip.permute(3, 0, 1, 2)  # (C, T, H, W)
            
            # 3. Subsample frames
            video_clip = subsample_frames(video_clip, clip_duration)
            
            # 4. Resize short side
            video_clip = resize_short_side_video(video_clip, 224)
            
            # 5. Normalize
            video_clip = normalize_video(
                video_clip,
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
            
            all_video.append(video_clip)
        
        # 6. Apply spatial crops
        all_video = SpatialCrop(224, num_crops=3)(all_video)
        
        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)
    
    return torch.stack(video_outputs, dim=0).to(device)
```

## Testing Strategy

1. **Create test data**: Small sample videos/audio
2. **Compare outputs**: Run both original and refactored versions
3. **Check shapes**: Ensure output tensors have identical shapes
4. **Validate values**: Check that values are close (within floating-point tolerance)
5. **Benchmark**: Measure if performance is acceptable

## Benefits of Refactoring

1. ✅ **Remove pytorchvideo dependency** - fixes the import error you're experiencing
2. ✅ **More control** - You understand every step of the pipeline
3. ✅ **Easier debugging** - No black-box library behavior
4. ✅ **Better compatibility** - Works with latest torchvision versions
5. ✅ **Lighter dependencies** - Fewer packages to manage

## Potential Issues

⚠️ **Decord installation**: May need special installation on some systems
⚠️ **Performance**: Custom implementation might be slower than optimized pytorchvideo
⚠️ **Edge cases**: Original library handles edge cases you might miss
⚠️ **Numerical differences**: Small differences in interpolation/sampling might affect results

## Next Steps

1. Start with audio refactoring (simpler)
2. Test thoroughly with known inputs
3. Move to video refactoring
4. Update any tests or downstream code
5. Remove pytorchvideo from requirements

Good luck with the refactoring! The code is now fully documented to help you understand each component.

