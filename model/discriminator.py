import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, text_embed_dim, image_channels=3):
        super(Discriminator, self).__init__()
        self.image_conv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * image_channels, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.joint = nn.Sequential(
            nn.Linear(512 + text_embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        img_feat = self.image_conv(image.view(image.size(0), -1))
        x = torch.cat((img_feat, text_embedding), dim=1)
        out = self.joint(x)
        return out
