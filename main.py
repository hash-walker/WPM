import tkinter as tk
import speech_recognition as sr 
import whisper
import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
import time
import threading
from pydub import AudioSegment

# Load the Whisper model
model = whisper.load_model('medium')
r = sr.Recognizer()

# Initialize Tkinter GUI
root = tk.Tk()
root.title("WPM Counter")
root.geometry("500x300")
root.configure(bg='black')

# Labels
wpm_label = tk.Label(root, text="WPM: 0", font=("Arial", 40), fg="white", bg="black")
wpm_label.pack(expand=True)
text_label = tk.Label(root, text="", font=("Arial", 14), fg="white", bg="black", wraplength=450)
text_label.pack(side=tk.BOTTOM, pady=20)

def update_ui(wpm, text):
    wpm_label.config(text=f"WPM: {wpm:.2f}")
    text_label.config(text=text)

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return (len(audio) / 1000) / 60  # Convert ms to minutes

def process_audio_chunk(audio_np, target_rate):
    filename = "audio.wav"
    wav_data = (audio_np * 32767).astype(np.int16)
    wavfile.write(filename, target_rate, wav_data)
    duration = get_audio_duration(filename)
    
    try:
        result = model.transcribe(filename)
        number_of_words = len(result["text"].split())
        wpm = number_of_words / duration
        update_ui(wpm, result["text"])
    except Exception as e:
        print("Error during transcription:", e)

# Speech recognition loop
def start_listening():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                audio = r.record(source, duration=7)
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_np = audio_data.astype(np.float32) / 32768.0
                original_rate = audio.sample_rate
                target_rate = 16000
                
                if original_rate != target_rate:
                    num_samples = round(len(audio_np) * target_rate / original_rate)
                    audio_np = scipy.signal.resample(audio_np, num_samples)
                
                threading.Thread(target=process_audio_chunk, args=(audio_np, target_rate)).start()
            except Exception as e:
                print("Error capturing audio:", e)

# Start the listening thread
t = threading.Thread(target=start_listening, daemon=True)
t.start()

root.mainloop()
