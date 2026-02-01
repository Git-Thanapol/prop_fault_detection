import os
import numpy as np
import librosa
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
from audiomentations import AddColorNoise, AddGaussianNoise, ApplyImpulseResponse, Gain
import warnings

import config

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class AudioAugmentor:
    def __init__(self):
        # 1. Initialize Torch Transforms (for visualization & some augs)
        self.spec_transform = T.Spectrogram(
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            window_fn=torch.hann_window,
            power=2.0
        )
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
            power=2.0
        )

        self.db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        # 2. Initialize Audiomentations
        self.aug_colored_noise = AddColorNoise(p=1.0, min_snr_db=15, max_snr_db=25)
        self.aug_gain = Gain(min_gain_db=-6, max_gain_db=6, p=1.0)
        self.aug_gaussian_noise = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        
        self.aug_rir = None
        if os.path.exists(config.RIR_PATH) and len(os.listdir(config.RIR_PATH)) > 0:
            # print(f"[INFO] Loaded RIR samples from {config.RIR_PATH}") # Reduce spam in multiprocessing
            try:
                self.aug_rir = ApplyImpulseResponse(ir_path=config.RIR_PATH, p=1.0)
            except Exception as e:
                print(f"[WARN] Failed to init RIR: {e}")
        else:
            pass # print(f"[WARN] No RIR files...") 

    def load_audio(self, filepath):
        """
        Load using Librosa (more robust than torchaudio on Windows).
        Ensures mono and correct sample rate.
        """
        # librosa loads as [samples,]
        try:
            y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
            return y
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

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
            return self.aug_colored_noise(samples=y_aug, sample_rate=config.SAMPLE_RATE)
        
        elif aug_type == "gain":
            return self.aug_gain(samples=y_aug, sample_rate=config.SAMPLE_RATE)
        
        elif aug_type == "gaussian_noise":
            return self.aug_gaussian_noise(samples=y_aug, sample_rate=config.SAMPLE_RATE)
            
        elif aug_type == "rir":
            if self.aug_rir:
                try:
                    return self.aug_rir(samples=y_aug, sample_rate=config.SAMPLE_RATE)
                except Exception as e:
                    # print(f"   [Error] RIR failed: {e}")
                    pass
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
            return F.pitch_shift(tensor, config.SAMPLE_RATE, n_steps)
        
        return tensor

    def apply_spectral_masking(self, spec_tensor, aug_type):
        """
        Applies masking directly to the spectrogram tensor.
        Randomizes mask width and number of masks for more variation.
        """
        if aug_type == "time_masking":
            # Apply 1 to 2 masks likely
            num_masks = np.random.randint(1, 4) 
            
            # Note: T.TimeMasking applies *one* mask of max length `time_mask_param`
            # Random position is handled by torchaudio internal logic (uniform sampling).
            masked_spec = spec_tensor
            for _ in range(num_masks):
                # Randomize MAX size for EACH mask
                t_param = np.random.randint(10, 40)
                masking = T.TimeMasking(time_mask_param=t_param)
                masked_spec = masking(masked_spec)
            return masked_spec
        
        elif aug_type == "frequency_masking":
            # Mask 10-20% of bins. 
            num_masks = np.random.randint(1, 3)
            
            masked_spec = spec_tensor
            for _ in range(num_masks):
                if spec_tensor.shape[-2] == config.N_MELS: # It's Mel
                    f_param = np.random.randint(10, 30) # Randomize param
                else: # It's Standard Spec
                    f_param = np.random.randint(50, 150)
                
                masking = T.FrequencyMasking(freq_mask_param=f_param)
                masked_spec = masking(masked_spec)
            return masked_spec
            
        return spec_tensor
