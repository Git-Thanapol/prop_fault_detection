import os
import glob
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, AddColorNoise, ApplyImpulseResponse, Gain
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_DIR = "../Samples"
OUTPUT_DIR = "dataset_output"

# External Assets Paths (Create these folders and put wav files inside if you want to use them)
RIR_PATH = "rir_samples"

# Audio Params
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 1024
N_MELS = 128
F_MIN = 20
F_MAX = 8000

# Augmentation List
AUGMENTATIONS = [
    "original",
    "polarity_inversion",
    "time_masking",
    "frequency_masking",
    "colored_noise",
    "gaussian_noise",  # Replaced background_noise
    "rir",
    "pitch_shift",
    "reverse",
    "gain"
]

class AudioPipeline:
    def __init__(self):
        # 1. Spectrogram Transform
        self.spec_transform = T.Spectrogram(
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            window_fn=torch.hann_window,
            power=2.0
        )
        
        # 2. MelSpectrogram Transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0
        )

        # 3. DB Converter
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        # 4. Define Audiomentations Pipelines
        # We initialize these conditionally to avoid errors if folders are missing
        self.aug_colored_noise = AddColorNoise(p=1.0, min_snr_db=15, max_snr_db=25)
        self.aug_gain = Gain(min_gain_db=-6, max_gain_db=6, p=1.0)
        
        # Gaussian Noise Setup (Replaces Background Noise)
        # Amplitudes are relative to signal (0.0 to 1.0). 0.015 is roughly perceptible hiss.
        self.aug_gaussian_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)

        # Check for RIR files
        if os.path.exists(RIR_PATH) and len(os.listdir(RIR_PATH)) > 0:
            self.aug_rir = ApplyImpulseResponse(ir_path=RIR_PATH, p=1.0)
        else:
            self.aug_rir = None
            print(f"[WARN] No RIR files found in '{RIR_PATH}'. Skipping RIR aug.")

    def load_audio(self, filepath):
        # Force "soundfile" backend to avoid torchcodec/ffmpeg issues on Windows
        # Ensure you have run: pip install soundfile
        waveform, sr = torchaudio.load(filepath, backend="soundfile")
        
        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        # Mix to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def save_image(self, tensor_db, output_path, cmap='viridis'):
        """
        Saves a tensor as a PNG image without axes.
        """
        # Ensure tensor is on CPU and numpy
        data = tensor_db.squeeze().numpy()
        
        # Flip Y-axis so low frequencies are at bottom
        data = np.flipud(data)

        # Setup plot
        plt.figure(figsize=(10, 5), frameon=False)
        # Create axes that fill the figure completely (no borders)
        ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)

        # Plot
        ax.imshow(data, aspect='auto', cmap=cmap)
        
        # Save
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def apply_augmentation(self, waveform, aug_type):
        """
        Applies waveform-level augmentations.
        Returns: Augmented Waveform (Tensor)
        """
        # Convert to numpy for audiomentations
        wav_np = waveform.numpy().squeeze()
        
        if aug_type == "original":
            return waveform
            
        elif aug_type == "polarity_inversion":
            return waveform * -1
            
        elif aug_type == "reverse":
            return torch.flip(waveform, [1])
            
        elif aug_type == "pitch_shift":
            # Shift between -2 and 2 semitones
            n_steps = np.random.uniform(-2, 2)
            return F.pitch_shift(waveform, SAMPLE_RATE, n_steps)
            
        elif aug_type == "gain":
            aug_np = self.aug_gain(samples=wav_np, sample_rate=SAMPLE_RATE)
            return torch.from_numpy(aug_np).unsqueeze(0)
            
        elif aug_type == "colored_noise":
            aug_np = self.aug_colored_noise(samples=wav_np, sample_rate=SAMPLE_RATE)
            return torch.from_numpy(aug_np).unsqueeze(0)
            
        elif aug_type == "gaussian_noise":
            aug_np = self.aug_gaussian_noise(samples=wav_np, sample_rate=SAMPLE_RATE)
            return torch.from_numpy(aug_np).unsqueeze(0)
                
        elif aug_type == "rir":
            if self.aug_rir:
                try:
                    aug_np = self.aug_rir(samples=wav_np, sample_rate=SAMPLE_RATE)
                    return torch.from_numpy(aug_np).unsqueeze(0)
                except:
                    return waveform
            else:
                return waveform # Skip
        
        # For Spectral masks, we return original waveform here, 
        # and handle masking in the visualization step
        elif aug_type in ["time_masking", "frequency_masking"]:
            return waveform

        return waveform

    def apply_spectral_augmentation(self, spec_tensor, aug_type):
        """
        Applies masking to the Spectrogram/MelSpectrogram Tensors.
        """
        if aug_type == "time_masking":
            # Mask 50-100 time steps
            masking = T.TimeMasking(time_mask_param=80)
            return masking(spec_tensor)
            
        elif aug_type == "frequency_masking":
            # Mask 10-20% of frequency bins (approx 20 bins for mel-128)
            masking = T.FrequencyMasking(freq_mask_param=20)
            return masking(spec_tensor)
            
        return spec_tensor

    def process_file(self, filepath):
        filename = os.path.basename(filepath).replace(".wav", "")
        print(f"Processing: {filename}")

        # 1. Load Base Audio
        base_waveform = self.load_audio(filepath)

        # 2. Iterate Augmentations
        for aug_type in AUGMENTATIONS:
            
            # A. Waveform Augmentation
            aug_waveform = self.apply_augmentation(base_waveform, aug_type)
            
            # --- Generate Spectrogram ---
            spec = self.spec_transform(aug_waveform)
            spec_db = self.db_transform(spec)
            
            # --- Generate Mel-Spectrogram ---
            mel = self.mel_transform(aug_waveform)
            mel_db = self.db_transform(mel)

            # B. Spectral Augmentation (if applicable)
            spec_db = self.apply_spectral_augmentation(spec_db, aug_type)
            mel_db = self.apply_spectral_augmentation(mel_db, aug_type)

            # 3. Save Images
            # Define Paths
            spec_dir = os.path.join(OUTPUT_DIR, "spectrograms", aug_type)
            mel_dir = os.path.join(OUTPUT_DIR, "mel_spectrograms", aug_type)
            
            os.makedirs(spec_dir, exist_ok=True)
            os.makedirs(mel_dir, exist_ok=True)

            # Save Spectrogram (Updated to 'viridis')
            self.save_image(spec_db, os.path.join(spec_dir, f"{filename}.png"), cmap='viridis')
            
            # Save Mel-Spectrogram (Updated to 'viridis')
            self.save_image(mel_db, os.path.join(mel_dir, f"{filename}.png"), cmap='viridis')

def main():
    # 1. Create Output Structure
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 2. Find Input Files
    audio_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}")
        return

    print(f"Found {len(audio_files)} audio files. Starting augmentation...")
    print(f"Output will be saved to: {OUTPUT_DIR}")

    # 3. Initialize Pipeline
    pipeline = AudioPipeline()

    # 4. Run Batch
    for filepath in audio_files:
        pipeline.process_file(filepath)

    print("\nBatch Processing Complete.")

if __name__ == "__main__":
    main()