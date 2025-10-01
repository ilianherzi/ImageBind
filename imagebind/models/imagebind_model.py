"""ImageBind multimodal embedding model.

This module implements the ImageBind model, which creates a joint embedding space
across six different modalities: vision, text, audio, thermal, depth, and IMU.
"""

from __future__ import annotations

import os
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn

from imagebind.models.helpers import (
    EinOpsRearrange,
    LearnableLogitScaling,
    Normalize,
    SelectElement,
    SelectEOSAndProject,
)
from imagebind.models.multimodal_preprocessors import (
    AudioPreprocessor,
    IMUPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
    ThermalPreprocessor,
)
from imagebind.models.transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class ImageBindModel(nn.Module):
    """ImageBind multimodal embedding model.

    This model creates a joint embedding space across multiple modalities including
    vision, text, audio, thermal images, depth maps, and IMU sensor data.

    Attributes:
        modality_preprocessors: ModuleDict containing preprocessing modules for each modality.
        modality_trunks: ModuleDict containing transformer trunks for each modality.
        modality_heads: ModuleDict containing projection heads for each modality.
        modality_postprocessors: ModuleDict containing postprocessing modules for each modality.
    """

    def __init__(
        self,
        video_frames: int = 2,
        kernel_size: tuple[int, int, int] = (2, 14, 14),
        audio_kernel_size: int = 16,
        audio_stride: int = 10,
        out_embed_dim: int = 768,
        vision_embed_dim: int = 1024,
        vision_num_blocks: int = 24,
        vision_num_heads: int = 16,
        audio_embed_dim: int = 768,
        audio_num_blocks: int = 12,
        audio_num_heads: int = 12,
        audio_num_mel_bins: int = 128,
        audio_target_len: int = 204,
        audio_drop_path: float = 0.1,
        text_embed_dim: int = 768,
        text_num_blocks: int = 12,
        text_num_heads: int = 12,
        depth_embed_dim: int = 384,
        depth_kernel_size: int = 16,
        depth_num_blocks: int = 12,
        depth_num_heads: int = 8,
        depth_drop_path: float = 0.0,
        thermal_embed_dim: int = 768,
        thermal_kernel_size: int = 16,
        thermal_num_blocks: int = 12,
        thermal_num_heads: int = 12,
        thermal_drop_path: float = 0.0,
        imu_embed_dim: int = 512,
        _imu_kernel_size: int = 8,
        imu_num_blocks: int = 6,
        imu_num_heads: int = 8,
        imu_drop_path: float = 0.7,
    ) -> None:
        """Initialize the ImageBind model.

        Args:
            video_frames: Number of frames for video inputs.
            kernel_size: 3D kernel size for vision convolution (temporal, height, width).
            audio_kernel_size: Kernel size for audio convolution.
            audio_stride: Stride for audio convolution.
            out_embed_dim: Output embedding dimension for all modalities.
            vision_embed_dim: Internal embedding dimension for vision transformer.
            vision_num_blocks: Number of transformer blocks for vision.
            vision_num_heads: Number of attention heads for vision.
            audio_embed_dim: Internal embedding dimension for audio transformer.
            audio_num_blocks: Number of transformer blocks for audio.
            audio_num_heads: Number of attention heads for audio.
            audio_num_mel_bins: Number of mel-frequency bins for audio.
            audio_target_len: Target length for audio spectrograms.
            audio_drop_path: Drop path rate for audio transformer.
            text_embed_dim: Internal embedding dimension for text transformer.
            text_num_blocks: Number of transformer blocks for text.
            text_num_heads: Number of attention heads for text.
            depth_embed_dim: Internal embedding dimension for depth transformer.
            depth_kernel_size: Kernel size for depth convolution.
            depth_num_blocks: Number of transformer blocks for depth.
            depth_num_heads: Number of attention heads for depth.
            depth_drop_path: Drop path rate for depth transformer.
            thermal_embed_dim: Internal embedding dimension for thermal transformer.
            thermal_kernel_size: Kernel size for thermal convolution.
            thermal_num_blocks: Number of transformer blocks for thermal.
            thermal_num_heads: Number of attention heads for thermal.
            thermal_drop_path: Drop path rate for thermal transformer.
            imu_embed_dim: Internal embedding dimension for IMU transformer.
            _imu_kernel_size: Kernel size for IMU preprocessing (unused, kept for API consistency).
            imu_num_blocks: Number of transformer blocks for IMU.
            imu_num_heads: Number of attention heads for IMU.
            imu_drop_path: Drop path rate for IMU transformer.
        """
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

    def _create_modality_preprocessors(
        self,
        video_frames: int = 2,
        vision_embed_dim: int = 1024,
        kernel_size: tuple[int, int, int] = (2, 14, 14),
        text_embed_dim: int = 768,
        audio_embed_dim: int = 768,
        audio_kernel_size: int = 16,
        audio_stride: int = 10,
        audio_num_mel_bins: int = 128,
        audio_target_len: int = 204,
        depth_embed_dim: int = 768,
        depth_kernel_size: int = 16,
        thermal_embed_dim: int = 768,
        thermal_kernel_size: int = 16,
        imu_embed_dim: int = 512,
    ) -> nn.ModuleDict:
        """Create preprocessing modules for each modality.

        Each preprocessor converts raw input data into tokenized representations
        with appropriate positional embeddings.

        Args:
            video_frames: Number of frames for video inputs.
            vision_embed_dim: Embedding dimension for vision.
            kernel_size: 3D kernel size for vision convolution.
            text_embed_dim: Embedding dimension for text.
            audio_embed_dim: Embedding dimension for audio.
            audio_kernel_size: Kernel size for audio convolution.
            audio_stride: Stride for audio convolution.
            audio_num_mel_bins: Number of mel-frequency bins.
            audio_target_len: Target length for audio spectrograms.
            depth_embed_dim: Embedding dimension for depth.
            depth_kernel_size: Kernel size for depth convolution.
            thermal_embed_dim: Embedding dimension for thermal.
            thermal_kernel_size: Kernel size for thermal convolution.
            imu_embed_dim: Embedding dimension for IMU.

        Returns:
            ModuleDict containing preprocessor modules for each modality.
        """
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=audio_embed_dim),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

        depth_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=depth_embed_dim),
        )

        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )

        thermal_stem = PatchEmbedGeneric(
            [
                nn.Conv2d(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=thermal_embed_dim),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )

        imu_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=48,
                    out_features=imu_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=imu_embed_dim),
        )

        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )

        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim: int = 1024,
        vision_num_blocks: int = 24,
        vision_num_heads: int = 16,
        text_embed_dim: int = 768,
        text_num_blocks: int = 12,
        text_num_heads: int = 12,
        audio_embed_dim: int = 768,
        audio_num_blocks: int = 12,
        audio_num_heads: int = 12,
        audio_drop_path: float = 0.0,
        depth_embed_dim: int = 768,
        depth_num_blocks: int = 12,
        depth_num_heads: int = 12,
        depth_drop_path: float = 0.0,
        thermal_embed_dim: int = 768,
        thermal_num_blocks: int = 12,
        thermal_num_heads: int = 12,
        thermal_drop_path: float = 0.0,
        imu_embed_dim: int = 512,
        imu_num_blocks: int = 6,
        imu_num_heads: int = 8,
        imu_drop_path: float = 0.7,
    ) -> nn.ModuleDict:
        """Create transformer trunk modules for each modality.

        Each trunk is a transformer encoder that processes the tokenized
        representations from the preprocessors.

        Args:
            vision_embed_dim: Embedding dimension for vision transformer.
            vision_num_blocks: Number of transformer blocks for vision.
            vision_num_heads: Number of attention heads for vision.
            text_embed_dim: Embedding dimension for text transformer.
            text_num_blocks: Number of transformer blocks for text.
            text_num_heads: Number of attention heads for text.
            audio_embed_dim: Embedding dimension for audio transformer.
            audio_num_blocks: Number of transformer blocks for audio.
            audio_num_heads: Number of attention heads for audio.
            audio_drop_path: Drop path rate for audio transformer.
            depth_embed_dim: Embedding dimension for depth transformer.
            depth_num_blocks: Number of transformer blocks for depth.
            depth_num_heads: Number of attention heads for depth.
            depth_drop_path: Drop path rate for depth transformer.
            thermal_embed_dim: Embedding dimension for thermal transformer.
            thermal_num_blocks: Number of transformer blocks for thermal.
            thermal_num_heads: Number of attention heads for thermal.
            thermal_drop_path: Drop path rate for thermal transformer.
            imu_embed_dim: Embedding dimension for IMU transformer.
            imu_num_blocks: Number of transformer blocks for IMU.
            imu_num_heads: Number of attention heads for IMU.
            imu_drop_path: Drop path rate for IMU transformer.

        Returns:
            ModuleDict containing transformer trunk modules for each modality.
        """

        def instantiate_trunk(
            embed_dim: int,
            num_blocks: int,
            num_heads: int,
            pre_transformer_ln: bool,
            add_bias_kv: bool,
            drop_path: float,
        ) -> SimpleTransformer:
            """Instantiate a transformer trunk with specified parameters.

            Args:
                embed_dim: Embedding dimension.
                num_blocks: Number of transformer blocks.
                num_heads: Number of attention heads.
                pre_transformer_ln: Whether to apply layer norm before transformer.
                add_bias_kv: Whether to add bias to key and value projections.
                drop_path: Drop path rate for stochastic depth.

            Returns:
                A configured SimpleTransformer module.
            """
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    (
                        nn.LayerNorm(embed_dim, eps=1e-6)
                        if pre_transformer_ln
                        else nn.Identity()
                    ),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim: int,
        vision_embed_dim: int,
        text_embed_dim: int,
        audio_embed_dim: int,
        depth_embed_dim: int,
        thermal_embed_dim: int,
        imu_embed_dim: int,
    ) -> nn.ModuleDict:
        """Create projection head modules for each modality.

        Each head projects the modality-specific embeddings to a common
        output dimension for cross-modal comparison.

        Args:
            out_embed_dim: Common output embedding dimension.
            vision_embed_dim: Input embedding dimension for vision.
            text_embed_dim: Input embedding dimension for text.
            audio_embed_dim: Input embedding dimension for audio.
            depth_embed_dim: Input embedding dimension for depth.
            thermal_embed_dim: Input embedding dimension for thermal.
            imu_embed_dim: Input embedding dimension for IMU.

        Returns:
            ModuleDict containing projection head modules for each modality.
        """
        modality_heads = {}

        modality_heads[ModalityType.VISION] = nn.Sequential(
            nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        modality_heads[ModalityType.AUDIO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.DEPTH] = nn.Sequential(
            nn.LayerNorm(normalized_shape=depth_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(depth_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.THERMAL] = nn.Sequential(
            nn.LayerNorm(normalized_shape=thermal_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(thermal_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.IMU] = nn.Sequential(
            nn.LayerNorm(normalized_shape=imu_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(imu_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, _out_embed_dim: int) -> nn.ModuleDict:
        """Create postprocessing modules for each modality.

        Each postprocessor normalizes the embeddings and applies modality-specific
        logit scaling for contrastive learning.

        Args:
            _out_embed_dim: Output embedding dimension (unused, kept for API consistency).

        Returns:
            ModuleDict containing postprocessor modules for each modality.
        """
        modality_postprocessors = {}

        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Processes inputs from multiple modalities and returns normalized embeddings
        in a joint space.

        Args:
            inputs: Dictionary mapping modality names to input tensors. Each tensor
                should have appropriate shape for its modality:
                - Vision: (B, C, T, H, W) or (B, S, C, T, H, W) for multiple clips
                - Text: (B, L) where L is sequence length
                - Audio: (B, C, H, W) or (B, S, C, H, W) for multiple clips
                - Depth: (B, C, H, W)
                - Thermal: (B, C, H, W)
                - IMU: (B, C, L)

        Returns:
            Dictionary mapping modality names to normalized embedding tensors
            of shape (B, out_embed_dim).
        """
        outputs = {}
        for modality_key, modality_value in inputs.items():
            # Audio and video inputs consist of multiple clips (ndim >= 5)
            reduce_list = modality_value.ndim >= 5
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                # Preprocess the input
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]

                # Process through transformer
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)

                # Project to common embedding space
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )

                # Normalize and scale
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )

                # Average over clips if multiple were provided
                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(dim=1)

                outputs[modality_key] = modality_value

        return outputs


def imagebind_huge(pretrained: bool = False) -> ImageBindModel:
    """Create an ImageBind model with 'huge' architecture.

    This variant uses larger embedding dimensions and more transformer blocks
    compared to the base configuration.

    Args:
        pretrained: If True, loads pretrained weights from Meta's repository.
            Weights will be downloaded to .checkpoints/imagebind_huge.pth if
            not already present.

    Returns:
        ImageBindModel configured with 'huge' architecture parameters, optionally
        loaded with pretrained weights.
    """
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        model.load_state_dict(
            torch.load(".checkpoints/imagebind_huge.pth", weights_only=True)
        )

    return model
