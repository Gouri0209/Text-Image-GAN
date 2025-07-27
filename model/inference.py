from models.generator import Generator
from models.text_encoder import TextEncoder
import torch
import matplotlib.pyplot as plt

G = Generator(100, 768)
G.load_state_dict(torch.load("generator.pth"))
G.eval()

text_encoder = TextEncoder()

def generate_image(prompt):
    text_embed = text_encoder.encode(prompt)
    noise = torch.randn(1, 100)
    with torch.no_grad():
        generated_img = G(noise, text_embed)
    img = generated_img.squeeze().permute(1, 2, 0).numpy()
    plt.imshow((img + 1) / 2)
    plt.axis("off")
    plt.show()

generate_image("a smiling person with glasses")
