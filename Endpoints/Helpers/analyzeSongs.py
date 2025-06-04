#Creamos funciones que nos facilitan el manejo de la funciones core o principales
import pyaudio

#Funcion que recibe abre el microfono y devuelve el audio grabado
def record_audio(duration=5, sample_rate=44100):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    
    print("Recording...")
    frames = []
    
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    return b''.join(frames)

#Funcion que recibe el audio grabado y lo guarda en un archivo wav
def save_audio_to_wav(audio_data, filename='recorded_audio.wav', sample_rate=44100):
    with open(filename, 'wb') as f:
        f.write(audio_data)
    print(f"Audio saved to {filename}")


#Funcion que recibe el audio grabado y lo analiza
def analyze_audio(audio_data, sample_rate=44100):
    # Placeholder for audio analysis logic
    # This could include feature extraction, model prediction, etc.
    print("Analyzing audio...")
    # For now, just return a dummy result
    return {
        "duration": len(audio_data) / sample_rate,
        "sample_rate": sample_rate,
        "message": "Audio analysis complete."
    }