import pandas as pd
import numpy as np
from scipy import signal
import os
import glob
import noisereduce as nr
import torch
import torchaudio

# --- CONFIGURATION ---
INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "dataset_processed"
NOISE_SAMPLE_FILE = "noise_sample.wav"  # Optional: Place a specific noise file in root
NORMALIZATION_THRESHOLD = 0.95          # Normalize peak to 95% of max volume

def get_lowpass_filter(sample_rate):
    """
    Design a Low Pass Filter:
    Passband: 700 Hz
    Stopband: 900 Hz
    """
    # Nyquist Frequency
    nyq = 0.5 * sample_rate
    
    # Normalized frequencies
    wp = 700 / nyq
    ws = 900 / nyq
    
    # Calculate order and cutoff for Butterworth filter
    N, Wn = signal.buttord(wp, ws, gpass=2, gstop=30)
    
    # Return Second-Order Sections (SOS) for stability
    sos = signal.butter(N, Wn, btype='low', output='sos')
    return sos

def normalize_audio(tensor):
    """
    Apply Peak Normalization using PyTorch operations.
    Scales the audio so the loudest point hits NORMALIZATION_THRESHOLD.
    Expects Tensor shape: [Channels, Time]
    """
    max_val = torch.max(torch.abs(tensor))
    
    # Avoid division by zero if the slice is absolute silence
    if max_val == 0:
        return tensor
    
    return tensor / max_val * NORMALIZATION_THRESHOLD

def process_all_metadata():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in dataset folder!")
        return

    print(f"Found {len(csv_files)} metadata files. Starting processing...\n")

    # Load external noise sample if it exists
    external_noise_profile = None
    if os.path.exists(NOISE_SAMPLE_FILE):
        print(f"Loading external noise sample: {NOISE_SAMPLE_FILE}")
        # Load with torchaudio (returns Tensor [Channels, Time], int SampleRate)
        noise_tensor, sr_noise = torchaudio.load(NOISE_SAMPLE_FILE)
        
        # Convert to numpy for noisereduce library (expects 1D or 2D array)
        # We assume mono for noise profile or take the first channel
        external_noise_profile = noise_tensor.numpy()
        if external_noise_profile.shape[0] == 1:
            external_noise_profile = external_noise_profile.squeeze()

    for csv_path in csv_files:
        process_single_csv(csv_path, external_noise_profile)

def process_single_csv(csv_path, external_noise_profile):
    print(f"--> Processing metadata: {os.path.basename(csv_path)}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        return

    for audio_filename, group in df.groupby("Audio_File"):
        full_audio_path = audio_filename
        if not os.path.exists(full_audio_path):
            print(f"   [WARNING] Audio file not found: {full_audio_path}. Skipping.")
            continue

        try:
            # Torchaudio loads as Float32 Tensor [Channels, Time]
            waveform, rate = torchaudio.load(full_audio_path)
        except Exception as e:
            print(f"   [ERROR] Could not read file {full_audio_path}: {e}")
            continue

        # Convert to numpy for Scipy filtering and Noisereduce
        # waveform is [C, T]. squeeze() makes it [T] if mono, which scipy/noisereduce prefer for 1D.
        # If stereo, we need to be careful. Let's process as numpy array.
        data_np = waveform.numpy()
        is_stereo = data_np.shape[0] > 1
        
        # Flatten for processing if mono to make library usage easier
        if not is_stereo:
            data_np = data_np.squeeze()

        # Prepare Filter
        sos_filter = get_lowpass_filter(rate)

        # Determine Noise Profile
        if external_noise_profile is not None:
            noise_clip = external_noise_profile
        else:
            # Fallback: Take first 0.5 seconds
            noise_end_idx = int(0.5 * rate)
            if is_stereo:
                noise_clip = data_np[:, 0:noise_end_idx]
            else:
                noise_clip = data_np[0:noise_end_idx]

        for index, row in group.iterrows():
            condition = str(row['Condition'])
            pwm = str(row['PWM'])
            iteration = str(row['Iter'])
            
            abs_start = row['Start_Abs']
            abs_end = row['End_Abs']
            ref_start = row['Audio_Start_Ref']

            start_offset_sec = abs_start - ref_start
            end_offset_sec = abs_end - ref_start

            start_idx = int(start_offset_sec * rate)
            end_idx = int(end_offset_sec * rate)

            # Safety bounds
            # data_np is either [Time] (mono) or [Channels, Time] (stereo)
            time_dim = data_np.shape[-1]
            start_idx = max(0, start_idx)
            end_idx = min(time_dim, end_idx)

            if start_idx >= end_idx:
                continue

            # 1. Extract Raw Slice (Numpy)
            if is_stereo:
                audio_slice = data_np[:, start_idx:end_idx]
            else:
                audio_slice = data_np[start_idx:end_idx]

            # 2. Apply Low Pass Filter (Scipy)
            # axis=-1 ensures we filter along time dimension even if stereo
            filtered_slice = signal.sosfilt(sos_filter, audio_slice, axis=-1)

            # 3. Apply Noise Reduction & Normalization (Two Branches)
            
            # --- Branch A: Stationary ---
            try:
                clean_stationary_np = nr.reduce_noise(
                    y=filtered_slice, 
                    sr=rate, 
                    y_noise=noise_clip, 
                    stationary=True
                )
                
                # Convert back to Torch Tensor for Normalization & Saving
                tensor_stat = torch.from_numpy(clean_stationary_np)
                if not is_stereo:
                    tensor_stat = tensor_stat.unsqueeze(0) # Ensure [C, T] for torchaudio

                # 4. Normalize (Torch)
                norm_stationary = normalize_audio(tensor_stat)

                # 5. Save (Torchaudio)
                save_wav(
                    f"{condition}_{pwm}_{iteration}_Stationary.wav", 
                    norm_stationary, 
                    rate
                )
            except Exception as e:
                print(f"   [Error] Stationary processing failed for {iteration}: {e}")

            # --- Branch B: Non-Stationary ---
            try:
                clean_nonstat_np = nr.reduce_noise(
                    y=filtered_slice, 
                    sr=rate, 
                    stationary=False
                )
                
                # Convert back to Torch Tensor
                tensor_nonstat = torch.from_numpy(clean_nonstat_np)
                if not is_stereo:
                    tensor_nonstat = tensor_nonstat.unsqueeze(0)

                # 4. Normalize (Torch)
                norm_nonstat = normalize_audio(tensor_nonstat)

                # 5. Save (Torchaudio)
                save_wav(
                    f"{condition}_{pwm}_{iteration}_Nonstationary.wav", 
                    norm_nonstat, 
                    rate
                )
            except Exception as e:
                print(f"   [Error] Non-stationary processing failed for {iteration}: {e}")

    print(f"   Done processing {os.path.basename(csv_path)}")

def save_wav(filename, tensor_data, rate):
    """
    Helper to save tensor using torchaudio.
    tensor_data should be [Channels, Time]
    """
    save_path = os.path.join(OUTPUT_FOLDER, filename)
    
    # Torchaudio handles saving float32 tensors automatically
    # It will save as 32-bit float WAV if input is float, or 16-bit PCM if int.
    # To mimic previous behavior (16-bit PCM), we can rely on torchaudio's encoding 
    # or just save as float (which is better for ML). 
    # Defaulting to float32 save for precision.
    
    torchaudio.save(save_path, tensor_data, rate)

if __name__ == "__main__":
    process_all_metadata()