import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, text_embed_dim, image_channels=3):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + text_embed_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64 * image_channels),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        out = self.fc(x)
        return out.view(-1, 3, 64, 64)
