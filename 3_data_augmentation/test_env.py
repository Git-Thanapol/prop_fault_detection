import sys
print(f"Python: {sys.version}")

try:
    print("Attempting to import torch...")
    import torch
    print(f"Torch Version: {torch.__version__}")
    
    print("Attempting to import torchaudio...")
    import torchaudio
    import torchaudio.transforms as T
    print(f"Torchaudio Version: {torchaudio.__version__}")
    
    # Test a simple transform initialization
    spec = T.Spectrogram(n_fft=2048)
    print("Torchaudio transform created successfully.")

except Exception as e:
    print("\n[IMPORT ERROR DETECTED]")
    print(e)
    sys.exit(1)
except OSError as e:
    print("\n[OS ERROR / SYMBOL LOOKUP FAILURE]")
    print(e)
    sys.exit(1)

print("\nEnvironment check passed!")
