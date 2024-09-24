from pydub import AudioSegment

# Load your audio file (mp3, wav, etc.)
audio = AudioSegment.from_file("input_audio_file.mp3")  # Replace with your file

# Convert to 16 kHz sample rate and export to wav format
audio = audio.set_frame_rate(16000)
audio = audio.set_channels(1)  # Ensure it's mono, as required by most ASR models
audio.export("output_audio_file.wav", format="wav")

print("Audio successfully converted and resampled to 16kHz")
