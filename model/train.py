from models.generator import Generator
from models.discriminator import Discriminator
from models.text_encoder import TextEncoder
import torch
import torch.nn as nn
import torch.optim as optim

# Define model configs
noise_dim = 100
text_embed_dim = 768  # BERT

# Instantiate models
G = Generator(noise_dim, text_embed_dim)
D = Discriminator(text_embed_dim)
text_encoder = TextEncoder()

# Define optimizers and loss
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(epochs):

    for text, real_image in data_loader:

        # Text → embedding
        text_embed = text_encoder.encode(text)

        # Random noise
        noise = torch.randn(
            real_image.size(0),
            noise_dim
        )

        # Generate fake image
        fake_image = G(noise, text_embed)


        # Train Discriminator


        optimizer_D.zero_grad()

        real_output = D(real_image, text_embed)
        fake_output = D(fake_image.detach(), text_embed)

        real_labels = torch.ones_like(real_output)
        fake_labels = torch.zeros_like(fake_output)

        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)

        D_loss = real_loss + fake_loss

        D_loss.backward()
        optimizer_D.step()

        # Train Generator

        optimizer_G.zero_grad()

        fake_output = D(fake_image, text_embed)

        G_loss = criterion(fake_output, real_labels)

        G_loss.backward()
        optimizer_G.step()
