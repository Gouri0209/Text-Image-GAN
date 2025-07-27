# Text-Image-GAN

A text-conditioned GAN that generates realistic 64x64 human face images from natural language prompts.

## ✨ Highlights
- Uses BERT (`dslim/bert-base-NER`) for embedding text prompts.
- Generator learns to synthesize face images conditioned on text.
- Discriminator evaluates both image realism and text-image alignment.
- Trained on custom base64-encoded images and prompt dataset in Parquet format.

## 🧠 Model Architecture
- Generator: Deep CNN with text conditioning.
- Discriminator: CNN with text-image fusion.
- Text Encoder: BERT-based contextual embeddings.

## 📁 Dataset
- Base64-encoded 64x64 human face images.
- Prompts extracted and embedded via `dslim/bert-base-NER`.

## 🛠️ How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Generate images
python generate.py
