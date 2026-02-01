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
# OPTIMIZED for F_MAX=1000Hz (Low Frequency Focus)
SAMPLE_RATE = 44100
# Increased N_FFT to 8192 to get ~5.4 Hz resolution per bin.
# (44100 / 8192 = 5.38 Hz).
# Standard 2048 would give ~21.5 Hz resolution, which is too coarse for 0-1000Hz.
N_FFT = 8192
HOP_LENGTH = 1024 # Overlap of 87.5% for smooth time steps
WIN_LENGTH = 8192 # Window size matching FFT
N_MELS = 128      # High vertical detail
F_MIN = 20
F_MAX = 1000

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
