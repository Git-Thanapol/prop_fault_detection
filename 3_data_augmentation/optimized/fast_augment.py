import os
import glob
import time
import argparse
import multiprocessing
import torch
import numpy as np
from tqdm import tqdm

# --- CRITICAL STABILITY FIXES ---
# Prevent libraries from spawning their own threads, which causes
# explosion when combined with multiprocessing (80 processes * N threads = crash).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import config
import augmentor
import visualizer

# Global hook for worker process
_augmentor_instance = None

def init_worker():
    """
    Initializes the augmentor in each worker process.
    Also ensures torch/numpy are restricted to single thread per process.
    """
    global _augmentor_instance
    
    # Double ensure threading limits inside the worker
    torch.set_num_threads(1)
    
    try:
        _augmentor_instance = augmentor.AudioAugmentor()
    except Exception as e:
        print(f"Worker init failed: {e}")

def process_file_wrapper(filepath):
    """
    Worker function to process a single file.
    """
    global _augmentor_instance
    if _augmentor_instance is None:
        _augmentor_instance = augmentor.AudioAugmentor()
    
    filename = os.path.basename(filepath).replace(".wav", "")
    
    try:
        # 1. Load Audio
        y_base = _augmentor_instance.load_audio(filepath)
        if y_base is None:
            return f"Failed to load {filename}"

        for aug_type in config.STRATEGIES:
            # 2. Waveform Augmentation
            y_aug = _augmentor_instance.apply_waveform_augmentation(y_base, aug_type)
            
            # 3. Convert to Tensor
            if not y_aug.flags['C_CONTIGUOUS']:
                y_aug = np.ascontiguousarray(y_aug)
            
            tensor_aug = torch.from_numpy(y_aug).float().unsqueeze(0)
            
            # 4. Tensor Augmentation
            if aug_type == "pitch_shift":
                tensor_aug = _augmentor_instance.apply_tensor_augmentation(tensor_aug, aug_type)
            
            # 5. Visualizations
            
            # --- SPECTROGRAM ---
            spec = _augmentor_instance.spec_transform(tensor_aug)
            spec_db = _augmentor_instance.db_transform(spec)
            
            if aug_type in ["time_masking", "frequency_masking"]:
                spec_db = _augmentor_instance.apply_spectral_masking(spec_db, aug_type)
                
            out_folder_spec = os.path.join(config.OUTPUT_DIR, "spectrograms")
            visualizer.save_visualization(
                spec_db,
                os.path.join(out_folder_spec, f"{filename}_{aug_type}.png")
            )
            
            # --- MEL SPECTROGRAM ---
            mel = _augmentor_instance.mel_transform(tensor_aug)
            mel_db = _augmentor_instance.db_transform(mel)
            
            if aug_type in ["time_masking", "frequency_masking"]:
                mel_db = _augmentor_instance.apply_spectral_masking(mel_db, aug_type)
                
            out_folder_mel = os.path.join(config.OUTPUT_DIR, "mel_spectrograms")
            visualizer.save_visualization(
                mel_db,
                os.path.join(out_folder_mel, f"{filename}_{aug_type}.png")
            )
            
        return None  # Success
        
    except Exception as e:
        return f"Error processing {filename}: {e}"

def main():
    multiprocessing.freeze_support()
    
    # Ensure output directories exist
    os.makedirs(os.path.join(config.OUTPUT_DIR, "spectrograms"), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "mel_spectrograms"), exist_ok=True)
    
    # Find files
    files = glob.glob(os.path.join(config.INPUT_DIR, "*.wav"))
    if not files:
        print(f"No wav files found in {config.INPUT_DIR}")
        return

    # Determine CPU count
    # Use 80% of cores to be safe, or all if confident in threading limits
    total_cores = os.cpu_count()
    max_workers = total_cores # Use all, trusting thread limits
    
    print(f"Found {len(files)} files.")
    print(f"Starting processing with {max_workers} processes...")
    
    start_time = time.time()
    
    # Use multiprocessing.Pool instead of Executor for maxtasksperchild support
    # maxtasksperchild=10 restarts workers every 10 files to free memory/resources
    with multiprocessing.Pool(processes=max_workers, initializer=init_worker, maxtasksperchild=10) as pool:
        # imap_unordered is often faster and allows smoother progress bars
        results = list(tqdm(pool.imap_unordered(process_file_wrapper, files), total=len(files), unit="file"))
        
    # Check results
    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"... and {len(errors)-10} more.")
    else:
        print("\nProcessing Complete successfully.")
        
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
