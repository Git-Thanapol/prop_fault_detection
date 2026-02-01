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
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 1024
N_MELS = 128
F_MIN = 20
F_MAX = 8000

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
