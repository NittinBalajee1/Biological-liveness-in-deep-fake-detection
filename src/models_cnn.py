from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class FrequencyBranch(nn.Module):
    def __init__(
        self,
        channels: list[int] | None = None,
        dropout: float = 0.35,
        input_channels: int = 2,
    ) -> None:
        super().__init__()
        channels = channels or [24, 48, 96]
        in_channels = input_channels
        blocks = []

        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, frequency_inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(frequency_inputs))


class SpatialBranch(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if backbone_name != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, spatial_inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(spatial_inputs)
        return self.projection(features)


class AttentionFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        spatial_logit_bias: float = 0.0,
        frequency_logit_bias: float = 0.0,
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.spatial_logit_bias = spatial_logit_bias
        self.frequency_logit_bias = frequency_logit_bias
        self.attention_temperature = max(attention_temperature, 1e-3)
        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, spatial_features: torch.Tensor, frequency_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([spatial_features, frequency_features], dim=1)
        scores = self.score(stacked)
        bias = scores.new_tensor([self.spatial_logit_bias, self.frequency_logit_bias]).view(1, 2, 1)
        scores = (scores + bias) / self.attention_temperature
        weights = torch.softmax(scores, dim=1)
        weighted = (stacked * weights).sum(dim=1)
        return weighted, weights.squeeze(-1)


class DualBranchDeepfakeDetector(nn.Module):
    def __init__(
        self,
        spatial_backbone: str = "resnet18",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        frequency_channels: list[int] | None = None,
        fusion_hidden_dim: int = 128,
        use_attention_fusion: bool = True,
        attention_hidden_dim: int = 64,
        dropout: float = 0.35,
        use_spatial: bool = True,
        use_fft: bool = True,
        use_dct: bool = True,
        spatial_logit_bias: float = 0.0,
        frequency_logit_bias: float = 0.0,
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_spatial = use_spatial
        self.use_fft = use_fft
        self.use_dct = use_dct
        self.frequency_indices = []
        if use_fft:
            self.frequency_indices.append(0)
        if use_dct:
            self.frequency_indices.append(1)
        self.use_frequency = bool(self.frequency_indices)

        if not self.use_spatial and not self.use_frequency:
            raise ValueError("At least one modality must be enabled.")

        self.spatial_branch = (
            SpatialBranch(
                backbone_name=spatial_backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
            )
            if self.use_spatial
            else None
        )
        self.frequency_branch = (
            FrequencyBranch(
                frequency_channels,
                dropout=dropout,
                input_channels=len(self.frequency_indices),
            )
            if self.use_frequency
            else None
        )
        self.use_attention_fusion = use_attention_fusion and self.use_spatial and self.use_frequency
        self.attention = (
            AttentionFusion(
                feature_dim=128,
                hidden_dim=attention_hidden_dim,
                spatial_logit_bias=spatial_logit_bias,
                frequency_logit_bias=frequency_logit_bias,
                attention_temperature=attention_temperature,
            )
            if self.use_attention_fusion
            else None
        )
        classifier_input_dim = 128
        if self.use_spatial and self.use_frequency and not self.use_attention_fusion:
            classifier_input_dim = 256
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        spatial_features = (
            self.spatial_branch(inputs["spatial"]) if self.use_spatial and self.spatial_branch else None
        )
        frequency_features = None
        if self.use_frequency and self.frequency_branch:
            frequency_input = inputs["frequency"][:, self.frequency_indices, :, :]
            frequency_features = self.frequency_branch(frequency_input)
        attention_weights = None
        if self.use_attention_fusion and self.attention is not None:
            fused, attention_weights = self.attention(spatial_features, frequency_features)
        elif self.use_spatial and self.use_frequency:
            fused = torch.cat([spatial_features, frequency_features], dim=1)
        elif self.use_spatial:
            fused = spatial_features
        else:
            fused = frequency_features
        logits = self.classifier(fused).squeeze(1)
        if return_attention:
            return logits, attention_weights
        return logits
