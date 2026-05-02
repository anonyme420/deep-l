"""
Model definitions:

ASTWithProjection — AST backbone + classification head + projection head.
The projection head outputs normalised embeddings used by PatchMixContrastiveLoss.
The classification head outputs logits used by FocalLoss.

EfficientNetModel  — EfficientNet-B3, fallback / ensemble partner.
EnsembleModel      — Weighted average of AST + EfficientNet probabilities.
"""

import io
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from src.config import NUM_CLASSES, IMG_SIZE, SAMPLE_RATE, DURATION, HOP_LENGTH


class ASTWithProjection(nn.Module):
    """
    Audio Spectrogram Transformer with:
      - Classification head  → 4-class logits (used with Focal Loss)
      - Projection head      → L2-normalised 128-D embeddings (used with Patch-Mix CL)

    The projection head is only used during training for the contrastive loss.
    At evaluation, only the classification head is active.
    """

    def __init__(
        self,
        num_classes:  int   = NUM_CLASSES,
        pretrained:   bool  = True,
        dropout:      float = 0.5,
        proj_dim:     int   = 128,
    ):
        super().__init__()
        assert TIMM_AVAILABLE, "pip install timm"

        # Backbone
        try:
            self.backbone = timm.create_model(
                "ast_patch16_224", pretrained=pretrained, num_classes=0
            )
            embed_dim = 768
            print("Using AST pretrained on AudioSet")
        except Exception:
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=pretrained, num_classes=0
            )
            embed_dim = self.backbone.embed_dim
            print("Fallback: ViT-base pretrained on ImageNet")

        self.embed_dim = embed_dim

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Projection head for contrastive learning (MLP: D → 256 → proj_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (before any head)."""
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: returns classification logits only."""
        return self.cls_head(self.encode(x))

    def forward_with_proj(self, x: torch.Tensor):
        """
        Returns (logits, projection_embedding).
        Used during Patch-Mix training steps.
        """
        feat = self.encode(x)
        logits = self.cls_head(feat)
        proj   = F.normalize(self.proj_head(feat), dim=1)
        return logits, proj


class PaSST_WithProjection(nn.Module):
    """
    PaSST (Patchout faSt Spectrogram Transformer) backbone with custom heads.

    Key advantages over AST:
      - AudioSet mAP 47.6% vs AST 34.4% (better pretrained features)
      - Built-in patchout augmentation (training=True) → reduces overfitting
        on small datasets like ICBHI without extra code

    Input:  (B, 1, 128, 998) AudioSet-normalised mel spectrogram
    Install: pip install hear21passt

    Architecture: DeiT-small backbone (embed_dim=384) + custom cls + proj heads
    """

    def __init__(
        self,
        num_classes: int   = NUM_CLASSES,
        pretrained:  bool  = True,
        dropout:     float = 0.5,
        proj_dim:    int   = 128,
    ):
        super().__init__()
        try:
            from hear21passt.base import get_model_passt
        except ImportError:
            raise ImportError(
                "PaSST not installed. Run:  pip install hear21passt"
            )

        # Load pretrained PaSST (keeps its original 527-class head; we bypass it)
        # Do NOT pass fstride/tstride — pretrained weights require the default
        # overlapping strides (fstride=10, tstride=10) used during AudioSet training.
        self.backbone = get_model_passt(
            arch="passt_s_swa_p16_128_ap476",
            pretrained=pretrained,
            n_classes=527,
            in_channels=1,
        )

        embed_dim = getattr(self.backbone, "embed_dim", 768)
        self.embed_dim = embed_dim

        # Resize time positional embedding from pretrained 10 s (99 patches) to
        # our 5 s clips (49 patches). This halves the attention sequence length
        # and gives ~4× speedup in the transformer blocks.
        self._resize_time_pos_embed()
        print(f"PaSST loaded | embed_dim={embed_dim} | pretrained={pretrained}")

        # Classification head (replaces PaSST's AudioSet head at inference)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Projection head for Patch-Mix contrastive loss
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def _resize_time_pos_embed(self):
        """Interpolate time positional embedding to match our 5 s clip length."""
        T_frames  = SAMPLE_RATE * DURATION // HOP_LENGTH           # 32000*5//320 = 500
        tstride   = self.backbone.patch_embed.proj.stride[1]        # 10
        T_patches = (T_frames - 16) // tstride + 1                  # = 49

        old        = self.backbone.time_new_pos_embed.data           # (1, D, 1, T_pre)
        T_pretrain = old.shape[-1]
        if T_pretrain == T_patches:
            return

        new_3d = F.interpolate(
            old.squeeze(2).float(), size=T_patches,
            mode="linear", align_corners=False,
        )
        self.backbone.time_new_pos_embed = nn.Parameter(
            new_3d.unsqueeze(2).to(old.dtype)                        # (1, D, 1, 49)
        )
        # Tell PatchEmbed the new expected size to silence the size-mismatch warning
        self.backbone.patch_embed.img_size = (128, T_frames)

        old_seq = 12 * T_pretrain + 2
        new_seq = 12 * T_patches  + 2
        print(f"  time pos embed: {T_pretrain}→{T_patches} patches | "
              f"sequence {old_seq}→{new_seq} tokens (~{old_seq**2 // new_seq**2}× attn speedup)")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run PaSST backbone; patchout fires automatically via self.training flag."""
        with contextlib.redirect_stdout(io.StringIO()):   # suppress hear21passt debug prints
            out = self.backbone(x)
        # PaSST returns (logits, feat) — feat is the pre-head CLS embedding
        feat = out[1] if isinstance(out, (tuple, list)) else out
        return feat   # (B, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.encode(x))

    def forward_with_proj(self, x: torch.Tensor):
        feat = self.encode(x)
        return self.cls_head(feat), F.normalize(self.proj_head(feat), dim=1)


class BEATs_WithProjection(nn.Module):
    """
    BEATs (Microsoft, AudioSet mAP 54.0%) + classification and projection heads.

    Input:  (B, 80000) raw 16 kHz waveform
    Setup on Kaggle:
        !git clone https://github.com/microsoft/unilm.git /kaggle/working/unilm
        !cp -r /kaggle/working/unilm/beats /kaggle/working/new_approach/beats
        # Download BEATs_iter3_plus_AS2M.pt to /kaggle/working/new_approach/
    """

    def __init__(
        self,
        num_classes:     int   = NUM_CLASSES,
        checkpoint_path: str   = None,
        dropout:         float = 0.5,
        proj_dim:        int   = 128,
    ):
        super().__init__()
        import sys, os
        beats_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "beats")
        if os.path.exists(beats_dir) and beats_dir not in sys.path:
            sys.path.insert(0, beats_dir)
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError:
            raise ImportError(
                "BEATs not found. Run:\n"
                "  !git clone https://github.com/microsoft/unilm.git /kaggle/working/unilm\n"
                "  !cp -r /kaggle/working/unilm/beats /kaggle/working/new_approach/beats"
            )

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "BEATs_iter3_plus_AS2M.pt",
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg  = BEATsConfig(ckpt["cfg"])
        self.backbone = BEATs(cfg)
        self.backbone.load_state_dict(ckpt["model"])

        embed_dim      = 768
        self.embed_dim = embed_dim
        print(f"BEATs loaded | embed_dim={embed_dim} | ckpt={checkpoint_path}")

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        feat, _      = self.backbone.extract_features(x, padding_mask=padding_mask)
        return feat.mean(dim=1)   # (B, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.encode(x))

    def forward_with_proj(self, x: torch.Tensor):
        feat = self.encode(x)
        return self.cls_head(feat), F.normalize(self.proj_head(feat), dim=1)


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        assert TIMM_AVAILABLE
        self.backbone = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0)
        in_features   = self.backbone.num_features
        self.head      = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class EnsembleModel(nn.Module):
    """Weighted softmax average of AST + EfficientNet."""
    def __init__(self, ast_model, eff_model, weight_ast: float = 0.65):
        super().__init__()
        self.ast        = ast_model
        self.eff        = eff_model
        self.weight_ast = weight_ast
        self.weight_eff = 1.0 - weight_ast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_ast = torch.softmax(self.ast(x), dim=1)
        p_eff = torch.softmax(self.eff(x), dim=1)
        return self.weight_ast * p_ast + self.weight_eff * p_eff


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "passt":
        return PaSST_WithProjection(pretrained=pretrained)
    elif name == "beats":
        return BEATs_WithProjection()
    elif name == "ast":
        return ASTWithProjection(pretrained=pretrained)
    elif name in ("efficientnet", "eff"):
        return EfficientNetModel(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
