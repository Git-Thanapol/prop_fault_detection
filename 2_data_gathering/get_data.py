import serial
import time
import logging
import threading
import csv
import wave
import pyaudio
import os
from dataclasses import dataclass
from datetime import datetime

# --- CONFIGURATION ---
# USB Serial Config
SERIAL_PORT = "COM3"  # <--- CHANGE THIS TO YOUR PORT (e.g., "COM3" or "/dev/ttyUSB0")
BAUD_RATE = 9600

# Audio Config
SAMPLE_RATE = 48000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_DEVICE_INDEX = None 

# Experiment Settings
RUN_DURATION = 6.0   # Time ON
COOLDOWN = 2.0       # Time OFF (Neutral)
BATCH_SIZE = 10      # Repeats per PWM
PWM_LIST = [1200, 1300, 1400, 1500, 1600, 1700, 1800]

# Logging
logging.basicConfig(format='%(asctime)s | %(message)s', level=logging.INFO, datefmt='%H:%M:%S')

class AsyncRecorder:
    """Records audio in background thread."""
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self.filename = ""
        self.start_timestamp = 0.0

    def start(self, filename):
        self.filename = filename
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop)
        self._thread.start()
        # Allow stream to initialize
        time.sleep(0.5) 
        return time.time()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _record_loop(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=CHANNELS,
                            rate=SAMPLE_RATE,
                            input=True,
                            input_device_index=AUDIO_DEVICE_INDEX,
                            frames_per_buffer=CHUNK_SIZE)
            
            frames = []
            logging.info(f"AUDIO: Recording started -> {self.filename}")
            
            while not self._stop_event.is_set():
                data = stream.read(CHUNK_SIZE)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            
            # Save File
            with wave.open(self.filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
            
            logging.info("AUDIO: Saved successfully.")
            
        except Exception as e:
            logging.error(f"AUDIO ERROR: {e}")
        finally:
            p.terminate()

class SerialController:
    def __init__(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            logging.info(f"Connected to {SERIAL_PORT}")
            # Wait for Arduino Auto-Reset
            time.sleep(2.0) 
            # Clear buffer
            self.ser.reset_input_buffer()
        except Exception as e:
            logging.error(f"Failed to connect to Serial: {e}")
            raise e

        self.recorder = AsyncRecorder()

    def send_cmd(self, command):
        """Sends command with newline character."""
        full_cmd = f"{command}\n"
        self.ser.write(full_cmd.encode('utf-8'))
        # Optional: Read acknowledgment to keep buffer clean
        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                # print(f"Arduino: {line}") # Uncomment for debugging
            except:
                pass

    def run_session(self, condition_name):
        # 1. Setup CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("dataset", exist_ok=True)
        csv_filename = f"dataset/Meta_{condition_name}_{timestamp_str}.csv"

        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Audio_File", "Condition", "PWM", "Iter", "Start_Abs", "End_Abs", "Audio_Start_Ref"])

            logging.info(f"--- STARTING SESSION: {condition_name} ---")

            # 2. Initialize Arduino
            self.send_cmd("stop")        # Safety reset
            self.send_cmd("activate 11") # Enable both motors
            self.send_cmd("start")       # Set running = true
            time.sleep(1)

            # 3. PWM Loop
            for pwm in PWM_LIST:
                audio_file = f"dataset/Audio_{condition_name}_{pwm}_{timestamp_str}.wav"
                
                # Start Long Record
                audio_start_ref = self.recorder.start(audio_file)
                time.sleep(1.0) 

                logging.info(f">>> Testing PWM: {pwm} <<<")

                for i in range(1, BATCH_SIZE + 1):
                    # Record precise start time
                    start_time = time.time()
                    
                    # MOTOR ON
                    self.send_cmd(f"set {pwm}")
                    
                    # Wait duration
                    time.sleep(RUN_DURATION)
                    
                    # MOTOR OFF (Neutral)
                    self.send_cmd("set 1500")
                    end_time = time.time()
                    
                    # Log
                    writer.writerow([audio_file, condition_name, pwm, i, start_time, end_time, audio_start_ref])
                    f.flush()
                    
                    print(f"   Iter {i}: Done.")
                    
                    # Cooldown
                    time.sleep(COOLDOWN)

                # Stop Audio
                self.recorder.stop()
                time.sleep(1.0)

            # 4. Finish
            self.send_cmd("stop")
            logging.info("--- SESSION COMPLETE ---")

    def close(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.send_cmd("stop")
            self.ser.close()

if __name__ == "__main__":
    controller = None
    try:
        controller = SerialController()
        
        # --- INPUT YOUR CONDITION HERE ---
        # e.g., "Healthy", "Imbalance_L", "Barnacle_R"
        current_condition = "Healthy" 
        
        controller.run_session(current_condition)
        
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if controller:
            controller.close()