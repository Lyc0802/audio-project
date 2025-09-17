# Project Description

This project uses Google’s **NSynth** musical instrument sound dataset, converting audio into **Mel spectrograms** and building a model with a **Variational Autoencoder (VAE)** to learn latent representations of timbre. By applying **Interpolation** and **Latent arithmetic**, it enables smooth transitions between instruments and manipulation of timbral characteristics to generate new sounds.

---

## Key Terminologies

### Mel spectrogram  
A **Mel spectrogram** is a two-dimensional representation of audio (time × frequency). Its frequency axis is transformed using the **Mel scale**, which aligns with human perception of pitch: providing finer resolution at lower frequencies and coarser resolution at higher frequencies. Compared to raw waveforms, Mel spectrograms are more suitable for machine learning models to capture perceptual sound features.

### Variational Autoencoder (VAE)  
A **VAE** is a generative model composed of an **encoder** and a **decoder**:  
- The encoder compresses an input Mel spectrogram into a **latent vector**, learning its probability distribution (mean and variance).  
- Through the **reparameterization trick**, random sampling is possible during training while preserving gradient flow.  
- The decoder reconstructs the Mel spectrogram from the latent vector.  

The latent space learned by a VAE is continuous and smooth, making it well-suited for sound generation and timbre transformation.

### Interpolation  
**Interpolation** refers to taking intermediate values between two latent vectors in latent space. For example, interpolating between a piano vector and a violin vector produces a sequence of sounds that gradually transform from piano to violin. This enables **smooth timbral morphing** between instruments.

### Latent arithmetic  
**Latent arithmetic** is the manipulation of vectors directly in latent space to modify specific sound attributes. For example:  
- (Violin + Brightness) − (Darkness) ≈ A brighter violin sound  
- (Piano − Original) + Synth ≈ A piano sound with synthesizer-like qualities  

This technique leverages the structured nature of the latent space, where certain semantic attributes correspond to approximately linear directions.

---

## Conclusion  
By combining Mel spectrograms and Variational Autoencoders, this project explores the structure of timbre in a continuous latent space. Using interpolation and latent arithmetic, it demonstrates how machine learning can generate new, perceptually meaningful sounds and provide intuitive control over timbral transformations.
