

# Multimodal Speech Recognition with Lip Reading

This project is a **multimodal speech recognition system** that combines **audio-based speech recognition** using a pre-trained `Wav2Vec2` model from Hugging Face, and **visual lip-reading** based on a 3D convolutional neural network (Conv3D). The system is designed to improve speech recognition accuracy by incorporating both audio and video (lip movements) inputs.

## Features

- **Audio-based Speech Recognition**: Utilizes the `Wav2Vec2` model from Hugging Face for automatic speech recognition (ASR).
- **Lip-Reading**: Processes video input (lip movements) using a Conv3D network to extract visual features that aid in the speech recognition process.
- **Multimodal Fusion**: Combines audio and visual data to improve the overall recognition accuracy.
  
## Requirements

### System Requirements
- **CUDA 12.x** (for GPU acceleration)
- **cuDNN 9.x** (for TensorFlow GPU support)
  
### Software Dependencies

1. Python 3.8+
2. TensorFlow 2.13.1
3. Hugging Face `transformers` library
4. `MoviePy` for video processing
5. `FFmpeg` for handling multimedia files

To install the dependencies, you can run the following:

```bash
pip install -r requirements.txt
```

Where `requirements.txt` includes:
```
tensorflow==2.13.1
transformers==4.30.0
moviepy
```

### Additional Setup
Make sure to install FFmpeg on your system:

```bash
sudo apt-get install ffmpeg
```

## Project Structure

- `multimodal_speech_recognition.py`: Main script that builds and runs the multimodal speech recognition model.
- `extract_video_frames.py`: Extracts video frames from a video for lip-reading.
- `speech_recognition.py`: Performs speech recognition on an audio file.
- `convert_audio.py`: Converts the audio input to the required format for processing.
- `README.md`: This file, which provides an overview of the project.
  
## How to Run the Project

### 1. Prepare the Input

First, you will need an **audio-visual dataset** containing both speech and the corresponding video of the speaker.

To download a YouTube video, you can use `yt-dlp`:

```bash
yt-dlp -f bestvideo+bestaudio --merge-output-format mp4 "<YouTube Video URL>" -o "speech_video.mp4"
```

### 2. Extract Audio and Video Frames

You can extract audio and video frames using the provided scripts:

```bash
python extract_video_frames.py --input speech_video.mp4
python convert_audio.py --input speech_video.mp4
```

### 3. Run the Multimodal Speech Recognition

To run the multimodal speech recognition system, execute:

```bash
python multimodal_speech_recognition.py
```

This will load the model, process the audio and video inputs, and output the recognized text.

### 4. Fine-tuning (Optional)

To improve the performance, you can fine-tune the `Wav2Vec2` model using your own dataset. Check Hugging Face's [fine-tuning guide](https://huggingface.co/transformers/training.html) for more details.

## Future Work

- **Fine-tuning the Model**: Fine-tuning the `Wav2Vec2` model on a custom dataset with multimodal input.
- **Improved Video Processing**: Enhancing the video processing pipeline to work with real-time video input.
- **End-to-End Integration**: Developing an end-to-end system that can be used in real-time applications such as video conferencing.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributions

Contributions are welcome! Feel free to submit a pull request or open an issue if you encounter any problems.

---


