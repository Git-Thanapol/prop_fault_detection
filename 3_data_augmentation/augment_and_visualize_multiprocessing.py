import os
import glob
import numpy as np
import librosa
import matplotlib
# Fix for Tcl/Tk errors: Use 'Agg' backend for non-interactive image saving
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from audiomentations import Compose, AddColorNoise, AddGaussianNoise, ApplyImpulseResponse, Gain
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_output")

# External Assets Paths
RIR_PATH = os.path.join(PROJECT_ROOT, "rir_samples")

# Audio Params
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 1024
N_MELS = 128
F_MIN = 20
F_MAX = 8000

# Image Output Params
FIG_SIZE = (3.84, 3.84) 

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
        # Initialize Audiomentations
        self.aug_colored_noise = AddColorNoise(p=1.0, min_snr_db=15, max_snr_db=25)
        self.aug_gain = Gain(min_gain_db=-6, max_gain_db=6, p=1.0)
        self.aug_gaussian_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        
        self.aug_rir = None
        # Note: In multiprocessing, checking file existence here might emit multiple logs
        if os.path.exists(RIR_PATH) and len(os.listdir(RIR_PATH)) > 0:
            self.aug_rir = ApplyImpulseResponse(ir_path=RIR_PATH, p=1.0)

    def load_audio(self, filepath):
        try:
            y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
            return y
        except Exception as e:
            return None

    def apply_waveform_augmentation(self, y, aug_type):
        y_aug = y.copy()
        if aug_type == "original": return y_aug
        elif aug_type == "polarity_inversion": return -1 * y_aug
        elif aug_type == "reverse": return np.flip(y_aug)
        elif aug_type == "colored_noise": return self.aug_colored_noise(samples=y_aug, sample_rate=SAMPLE_RATE)
        elif aug_type == "gain": return self.aug_gain(samples=y_aug, sample_rate=SAMPLE_RATE)
        elif aug_type == "gaussian_noise": return self.aug_gaussian_noise(samples=y_aug, sample_rate=SAMPLE_RATE)
        elif aug_type == "rir":
            if self.aug_rir:
                try: return self.aug_rir(samples=y_aug, sample_rate=SAMPLE_RATE)
                except: pass
            return y_aug
        elif aug_type == "pitch_shift":
            n_steps = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(y_aug, sr=SAMPLE_RATE, n_steps=n_steps)
        return y_aug

    def apply_spectral_masking_numpy(self, spec_db, aug_type, is_mel=False):
        spec_masked = spec_db.copy()
        n_freqs, n_time = spec_masked.shape
        min_val = spec_masked.min()

        if aug_type == "time_masking":
            T_param = 10
            t = np.random.randint(0, T_param + 1)
            t0 = np.random.randint(0, n_time - t + 1)
            if t > 0: spec_masked[:, t0:t0+t] = min_val
        
        elif aug_type == "frequency_masking":
            F_param = 20 if is_mel else 100
            f = np.random.randint(0, F_param + 1)
            f0 = np.random.randint(0, n_freqs - f + 1)
            if f > 0: spec_masked[f0:f0+f, :] = min_val
            
        return spec_masked

    def save_visualization(self, data, output_path, cmap='viridis'):
        plt.figure(figsize=FIG_SIZE, frameon=False)
        ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def process_file_logic(self, filepath):
        """
        Worker function logic.
        """
        filename = os.path.basename(filepath).replace(".wav", "")
        # print(f"Processing: {filename}") # Avoid printing in MP 

        y_base = self.load_audio(filepath)
        if y_base is None:
            return False

        for aug_type in AUGMENTATIONS:
            y_aug = self.apply_waveform_augmentation(y_base, aug_type)

            # --- SPECTROGRAM ---
            D = librosa.stft(y_aug, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
            spec = np.abs(D)**2
            spec_db = librosa.power_to_db(spec, ref=np.max)
            
            if aug_type in ["time_masking", "frequency_masking"]:
                spec_db = self.apply_spectral_masking_numpy(spec_db, aug_type, is_mel=False)
            
            out_folder_spec = os.path.join(OUTPUT_DIR, "spectrograms")
            os.makedirs(out_folder_spec, exist_ok=True)
            self.save_visualization(spec_db, os.path.join(out_folder_spec, f"{filename}_{aug_type}.png"))

            # --- MEL SPECTROGRAM ---
            mel = librosa.feature.melspectrogram(
                y=y_aug, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                win_length=WIN_LENGTH, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            if aug_type in ["time_masking", "frequency_masking"]:
                mel_db = self.apply_spectral_masking_numpy(mel_db, aug_type, is_mel=True)

            out_folder_mel = os.path.join(OUTPUT_DIR, "mel_spectrograms")
            os.makedirs(out_folder_mel, exist_ok=True)
            self.save_visualization(mel_db, os.path.join(out_folder_mel, f"{filename}_{aug_type}.png"))
        
        return True

# --- WORKER WRAPPER ---
# Global instance for workers (optional optimization to avoid recreating per call if using fork)
_augmentor = None

def init_worker():
    """Initialize the augmentor once per worker process."""
    global _augmentor
    _augmentor = AudioAugmentor()

def process_file_wrapper(filepath):
    """Wrapper that uses the global augmentor instance."""
    global _augmentor
    if _augmentor is None:
        _augmentor = AudioAugmentor() # Fallback
    return _augmentor.process_file_logic(filepath)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))
    if not files:
        print(f"No wav files found in {INPUT_DIR}")
        return

    num_cores = os.cpu_count()
    print(f"Found {len(files)} files in {INPUT_DIR}.")
    print(f"Starting multiprocessing with {num_cores} cores...")

    # Use max_workers=None to default to os.cpu_count()
    # Using initializer to setup objects once per process
    with ProcessPoolExecutor(max_workers=num_cores, initializer=init_worker) as executor:
        # Submit all tasks
        results = list(tqdm(executor.map(process_file_wrapper, files), total=len(files)))

    print("\nProcessing Complete.")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
