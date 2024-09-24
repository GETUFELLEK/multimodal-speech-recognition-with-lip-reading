from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
import tensorflow as tf
import soundfile as sf

# Load pre-trained model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio file and prepare input
def load_audio(file_path):
    speech, sample_rate = sf.read(file_path)
    return speech, sample_rate

# Update with the correct path to your audio file
file_path = "/home/getu/projects_2024/multimodal-ai/output_audio_file.wav"

# Load and process input
speech, sample_rate = load_audio(file_path)

# Prepare input for the model
inputs = processor(speech, sampling_rate=sample_rate, return_tensors="tf", padding=True)

# Perform inference
logits = model(inputs.input_values).logits
predicted_ids = tf.argmax(logits, axis=-1)

# Decode prediction
transcription = processor.batch_decode(predicted_ids)
print("Transcription: ", transcription)
