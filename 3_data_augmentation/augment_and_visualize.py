import os
import glob
import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import matplotlib
# Fix for Tcl/Tk errors: Use 'Agg' backend for non-interactive image saving
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from audiomentations import Compose, AddColorNoise, AddGaussianNoise, ApplyImpulseResponse, Gain
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_DIR = "../Samples"
OUTPUT_DIR = "dataset_output"

# External Assets Paths
# Create these folders and place .wav files inside for these augmentations to work
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
    "gaussian_noise",
    "rir",
    "pitch_shift",
    "reverse",
    "gain"
]

class AudioAugmentor:
    def __init__(self):
        # 1. Initialize Torch Transforms (for visualization & some augs)
        self.spec_transform = T.Spectrogram(
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            window_fn=torch.hann_window,
            power=2.0
        )
        
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

        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        # 2. Initialize Audiomentations
        self.aug_colored_noise = AddColorNoise(p=1.0, min_snr_db=15, max_snr_db=25)
        self.aug_gain = Gain(min_gain_db=-6, max_gain_db=6, p=1.0)
        self.aug_gaussian_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        
        self.aug_rir = None
        if os.path.exists(RIR_PATH) and len(os.listdir(RIR_PATH)) > 0:
            print(f"[INFO] Loaded RIR samples from {RIR_PATH}")
            self.aug_rir = ApplyImpulseResponse(ir_path=RIR_PATH, p=1.0)
        else:
            print(f"[WARN] No RIR files in '{RIR_PATH}'. 'rir' aug will be skipped.")

    def load_audio(self, filepath):
        """
        Load using Librosa (more robust than torchaudio on Windows).
        Ensures mono and correct sample rate.
        """
        # librosa loads as [samples,]
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return y

    def apply_waveform_augmentation(self, y, aug_type):
        """
        Applies augmentation to the raw waveform (numpy array).
        Returns: Augmented numpy array
        """
        # Safety copy
        y_aug = y.copy()

        if aug_type == "original":
            return y_aug
        
        elif aug_type == "polarity_inversion":
            return -1 * y_aug
        
        elif aug_type == "reverse":
            # FIXED: Added .copy() because np.flip returns an array with negative strides,
            # which torch.from_numpy() cannot handle.
            return np.flip(y_aug).copy()
        
        elif aug_type == "colored_noise":
            return self.aug_colored_noise(samples=y_aug, sample_rate=SAMPLE_RATE)
        
        elif aug_type == "gain":
            return self.aug_gain(samples=y_aug, sample_rate=SAMPLE_RATE)
        
        elif aug_type == "gaussian_noise":
            return self.aug_gaussian_noise(samples=y_aug, sample_rate=SAMPLE_RATE)
            
        elif aug_type == "rir":
            if self.aug_rir:
                try:
                    return self.aug_rir(samples=y_aug, sample_rate=SAMPLE_RATE)
                except Exception as e:
                    print(f"   [Error] RIR failed: {e}")
            return y_aug

        # For spectral/tensor augmentations (PitchShift, TimeMask, FreqMask), 
        # we return original here and handle them later or via Tensor conversion.
        return y_aug

    def apply_tensor_augmentation(self, tensor, aug_type):
        """
        Applies augmentations that require PyTorch Tensors (Pitch Shift).
        """
        if aug_type == "pitch_shift":
            # Pitch shift -2 to +2 semitones
            n_steps = np.random.uniform(-2, 2)
            return F.pitch_shift(tensor, SAMPLE_RATE, n_steps)
        
        return tensor

    def apply_spectral_masking(self, spec_tensor, aug_type):
        """
        Applies masking directly to the spectrogram tensor.
        """
        if aug_type == "time_masking":
            # Mask 50-100 ms. 
            # We need to convert ms to frames. 
            # 1 frame = HOP_LENGTH / SAMPLE_RATE = 512 / 48000 ≈ 10.6ms
            # 50ms ≈ 5 frames, 100ms ≈ 10 frames
            masking = T.TimeMasking(time_mask_param=10)
            return masking(spec_tensor)
        
        elif aug_type == "frequency_masking":
            # Mask 10-20% of bins. 
            # For Spectrogram (N_FFT=2048 -> 1025 bins), 10% is ~100.
            # For Mel (128 bins), 10% is ~13.
            # We'll use a conservative param that works for both or check dimensions.
            if spec_tensor.shape[-2] == N_MELS: # It's Mel
                masking = T.FrequencyMasking(freq_mask_param=20)
            else: # It's Standard Spec
                masking = T.FrequencyMasking(freq_mask_param=100)
            return masking(spec_tensor)
            
        return spec_tensor

    def save_visualization(self, spec_tensor, output_path, cmap='viridis'):
        """
        Saves the tensor as a PNG image without axes.
        """
        # Move to CPU numpy
        data = spec_tensor.squeeze().numpy()
        
        # Origin is normally bottom-left for spec, but imshow expects image data convention.
        # Librosa/Matplotlib specs usually need origin='lower' in imshow.
        # Or flipud manually.
        
        plt.figure(figsize=(10, 5), frameon=False)
        ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)

        # origin='lower' puts low frequencies at the bottom
        ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def process_file(self, filepath):
        filename = os.path.basename(filepath).replace(".wav", "")
        print(f"Processing: {filename}")

        # 1. Load Audio (Numpy)
        y_base = self.load_audio(filepath)

        for aug_type in AUGMENTATIONS:
            # 2. Apply Waveform Augmentation (Numpy)
            y_aug = self.apply_waveform_augmentation(y_base, aug_type)

            # 3. Convert to Tensor [1, Time]
            # Ensure array is contiguous and has positive strides before conversion
            if not y_aug.flags['C_CONTIGUOUS']:
                y_aug = np.ascontiguousarray(y_aug)
                
            tensor_aug = torch.from_numpy(y_aug).float().unsqueeze(0)

            # 4. Apply Tensor Augmentation (Pitch Shift)
            if aug_type == "pitch_shift":
                tensor_aug = self.apply_tensor_augmentation(tensor_aug, aug_type)

            # 5. Generate Visualizations
            
            # --- SPECTROGRAM ---
            spec = self.spec_transform(tensor_aug)
            spec_db = self.db_transform(spec)
            
            # Apply Masking (if selected)
            if aug_type in ["time_masking", "frequency_masking"]:
                spec_db = self.apply_spectral_masking(spec_db, aug_type)
            
            # Save
            out_folder_spec = os.path.join(OUTPUT_DIR, "spectrograms", aug_type)
            os.makedirs(out_folder_spec, exist_ok=True)
            self.save_visualization(spec_db, os.path.join(out_folder_spec, f"{filename}.png"), cmap='viridis')

            # --- MEL SPECTROGRAM ---
            mel = self.mel_transform(tensor_aug)
            mel_db = self.db_transform(mel)
            
            # Apply Masking (if selected)
            if aug_type in ["time_masking", "frequency_masking"]:
                mel_db = self.apply_spectral_masking(mel_db, aug_type)

            # Save
            out_folder_mel = os.path.join(OUTPUT_DIR, "mel_spectrograms", aug_type)
            os.makedirs(out_folder_mel, exist_ok=True)
            self.save_visualization(mel_db, os.path.join(out_folder_mel, f"{filename}.png"), cmap='viridis')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find Files
    files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not files:
        print(f"No wav files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} files. Starting...")
    
    augmentor = AudioAugmentor()

    for f in files:
        augmentor.process_file(f)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()