import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

# from mamba_block.model import MambaModule

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=128):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 192, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(192)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size=5, emb_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, emb_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)  # shape: (b, num_patches, emb_size)

class PlantXMamba(nn.Module):
    def __init__(self, num_classes=4, patch_size=5, emb_size=16, d_state=64,d_conv=64,expand=4,n_layers=2,num_blocks=4, dropout=0.1):
        super().__init__()

        # VGG16 (2 blocks)
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_block = nn.Sequential(*list(vgg.features[:10]))

        # Inception-like block → (B, 512, 56, 56)
        self.inception = InceptionBlock(in_channels=128)

        # Patch Embedding → (B, 121, 16)
        self.patch_embed = PatchEmbedding(in_channels=512, patch_size=patch_size, emb_size=emb_size)

        # Mamba blocks
        mamba_args = type('Args', (), {
            'd_model': emb_size,
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'n_layers': n_layers,
            'dropout': dropout
        })()
        self.mamba = nn.Sequential(*[MambaModule(mamba_args) for _ in range(num_blocks)])

        # Classification head
        self.norm = nn.LayerNorm(emb_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.vgg_block(x)  # (B, 128, 56, 56)
        x = self.inception(x)  # (B, 512, 56, 56)
        x = self.patch_embed(x)  # (B, 121, 16)
        x = self.mamba(x)  # (B, 121, 16)
        x = self.norm(x)  # (B, 121, 16)
        x = x.permute(0, 2, 1)  # (B, 16, 121)
        x = self.global_pool(x).squeeze(-1)  # (B, 16)
        return self.classifier(x)  # (B, num_classes)

# Kiểm tra mô hình
if __name__ == "__main__":
    model = PlantXMamba(num_classes=4, patch_size=5, emb_size=16, num_blocks=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Nên là (1, 4)