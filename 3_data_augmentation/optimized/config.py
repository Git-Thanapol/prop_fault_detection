import os

# --- PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming structure: .../prop_fault_detection/3_data_augmentation/optimized/config.py
# So PROJECT_ROOT is two levels up: .../prop_fault_detection
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_output")
RIR_PATH = os.path.join(PROJECT_ROOT, "rir_samples")

# --- AUDIO PARAMS ---
# OPTIMIZED for SR=44100Hz, F_MAX=1200Hz
# Sweet spot calculation: Resolution = SR / N_FFT
# 44100 / 8192 = 5.38 Hz per bin.
# Maintains ~5.4 Hz resolution for 0-1200 Hz range.
SAMPLE_RATE = 44100
N_FFT = 8192
HOP_LENGTH = 1024    # Match time-step density (~23ms)
WIN_LENGTH = 8192    # Match FFT size
N_MELS = 128
F_MIN = 0
F_MAX = 1200

# --- VISUALIZATION PARAMS ---
# (Width, Height) in inches. Multiplied by dpi=100 gives pixel dimensions.
FIG_SIZE = (3.84, 3.84)
STRATEGIES = [
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
