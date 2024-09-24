import cv2
import tensorflow as tf
from tensorflow.keras import layers
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
import numpy as np
import moviepy.editor as mp

# Load pre-trained Wav2Vec2 model and processor from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Get the vocabulary size from the processor
vocab_size = len(processor.tokenizer)

# Define video processing model (for lip-reading)
video_input = tf.keras.Input(shape=(30, 128, 128, 3))  # Define a fixed sequence length of 30 frames

# Apply Conv3D and MaxPooling3D layers
x = layers.Conv3D(32, (3, 3, 3), activation='relu')(video_input)
x = layers.MaxPooling3D((2, 2, 2))(x)
x = layers.Conv3D(64, (3, 3, 3), activation='relu')(x)
x = layers.MaxPooling3D((2, 2, 2))(x)

# Flatten the output from the 3D convolutions
x = layers.Flatten()(x)

# Add a Dense layer for processing video features
video_output = layers.Dense(256, activation='relu')(x)

# Define the audio processing model using Wav2Vec2
audio_input = tf.keras.Input(shape=(None,))
audio_output = model(audio_input).logits  # Get the logits from the Wav2Vec2 model

# Apply Global Average Pooling to align the shape of audio_output
audio_output = layers.GlobalAveragePooling1D()(audio_output)

# Concatenate the outputs of the video and audio models
concatenated = layers.Concatenate()([audio_output, video_output])

# Add Dense layers for the final classification (speech recognition output)
x = layers.Dense(128, activation='relu')(concatenated)
x = layers.Dense(64, activation='relu')(x)
final_output = layers.Dense(vocab_size, activation='softmax')(x)

# Define and compile the multimodal model
multimodal_model = tf.keras.Model(inputs=[audio_input, video_input], outputs=final_output)
multimodal_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary to verify the architecture
multimodal_model.summary()

# Load and process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize and preprocess the frame for the model
        resized_frame = cv2.resize(frame, (128, 128))
        frames.append(resized_frame)
        
        if len(frames) >= 30:  # Example: Process 30 frames at a time
            break
    
    cap.release()
    
    # Convert the list of frames to a numpy array
    frames_array = tf.expand_dims(tf.convert_to_tensor(frames, dtype=tf.float32), axis=0)
    return frames_array

# Load and process audio
def process_audio(audio_path):
    video = mp.VideoFileClip(audio_path)
    audio = video.audio
    audio_samples = audio.to_soundarray(fps=16000)
    audio_samples = audio_samples.mean(axis=1)  # Convert stereo to mono
    return processor(audio_samples, sampling_rate=16000, return_tensors="tf").input_values

# Process the video and audio
video_frames = process_video("speech_video.mp4")
audio_data = process_audio("speech_video.mp4")

# Run inference with both video and audio data
predictions = multimodal_model([audio_data, video_frames])

# Decode predictions (as an example)
decoded_predictions = processor.batch_decode(tf.argmax(predictions, axis=-1))
print("Transcription:", decoded_predictions)
