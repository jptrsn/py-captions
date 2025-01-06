import os
import argparse
import whisper
from moviepy.editor import VideoFileClip
from webvtt import WebVTT, Caption

def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from video and saves it as a WAV file."""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def transcribe_audio_to_captions(audio_path, output_vtt_path, model_name="base"):
    """Transcribes audio using Whisper and generates VTT captions."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    webvtt = WebVTT()

    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Format start and end times for VTT format
        start_time_vtt = format_time(start_time)
        end_time_vtt = format_time(end_time)

        # Add caption to the VTT file
        caption = Caption(start_time_vtt, end_time_vtt, text)
        webvtt.captions.append(caption)

    # Save captions to a VTT file
    webvtt.save(output_vtt_path)
    print(f"VTT captions saved to {output_vtt_path}")

def format_time(seconds):
    """Formats time in seconds to VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"

# example command
# python3 script.py path/to/video.mp4 --output myfile.vtt --model large
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from an MP4 video file and generate VTT captions.")
    parser.add_argument("video_path", type=str, help="Path to the MP4 video file")
    parser.add_argument("--output", type=str, default="captions.vtt", help="Output VTT file path (default: captions.vtt)")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (e.g., tiny, base, small, medium, large)")

    args = parser.parse_args()
    video_path = args.video_path
    audio_path = "extracted_audio.wav"
    output_vtt_path = args.output
    model_name = args.model

    # Step 1: Extract audio from video
    extract_audio_from_video(video_path, audio_path)

    # Step 2: Transcribe audio and generate VTT file
    transcribe_audio_to_captions(audio_path, output_vtt_path, model_name)

    # Clean up extracted audio file if needed
    os.remove(audio_path)
