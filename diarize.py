import argparse
import os
import torch
import whisperx
from pyannote.audio import Pipeline
from pytube import YouTube
from pyannote.core import Segment
import math
from moviepy.editor import AudioFileClip
import soundfile as sf
import ssl
import urllib.request
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve the Hugging Face token from the environment
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if not huggingface_token:
    raise ValueError("Hugging Face token not found. Please set it in the .env file.")

import subprocess

def download_audio_from_youtube(url, output_path="temp_audio.wav"):
    print("Extracting audio from YouTube video using yt-dlp and ffmpeg...")

    yt_dlp_command = [
        "yt-dlp",
        "-f", "bestaudio",  # Best available audio format
        "--extract-audio",
        "--audio-format", "wav",
        "-o", output_path,  # Output file path
        url,
    ]

    try:
        subprocess.run(yt_dlp_command, check=True)
        print(f"Audio extracted and saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e.stderr.decode()}")
        raise RuntimeError("Failed to extract audio using yt-dlp")

    return output_path


def merge_segments_with_speakers(transcription_segments, diarization_result):
    """
    Merge transcription segments with speaker diarization data.
    :param transcription_segments: List of transcription segments with start and end times.
    :param diarization_result: Pyannote diarization result.
    :return: List of segments with speaker labels.
    """
    merged_segments = []
    for segment in transcription_segments:
        start, end = segment["start"], segment["end"]
        text = segment["text"]
        speaker = "Unknown"  # Default speaker label

        # Find the speaker with overlapping time
        for turn, _, label in diarization_result.itertracks(yield_label=True):
            if turn.start <= start <= turn.end or turn.start <= end <= turn.end:
                speaker = label
                break

        merged_segments.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text
        })
    return merged_segments

def diarize_with_progress(audio_path, chunk_duration=5):
    """
    Perform speaker diarization with progress updates by chunking the audio.

    :param audio_path: Path to the input audio file
    :param chunk_duration: Duration (in seconds) of each chunk
    :return: Merged diarization result
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=huggingface_token)

    # Get total duration of the audio file
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration
    audio_clip.close()

    num_chunks = math.ceil(total_duration / chunk_duration)
    print(f"Processing audio in {num_chunks} chunks...")

    diarization_result = None
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)

        # Process the current chunk
        print(f"Processing chunk {i + 1}/{num_chunks} ({start_time:.2f}s - {end_time:.2f}s)...")
        current_segment = Segment(start_time, end_time)

        # Prepare input with cropped segment
        input_data = {"audio": audio_path, "uri": f"chunk_{i+1}", "duration": chunk_duration, "segment": current_segment}

        # Perform diarization
        current_result = pipeline(input_data)

        # Merge results
        if diarization_result is None:
            diarization_result = current_result
        else:
            diarization_result.update(current_result)

    print("Diarization complete!")
    return diarization_result

def transcribe_with_diarization(input_file, output_file="output.ass"):
    """
    Transcribe a video or audio file with speaker diarization using WhisperX.

    :param input_file: Path to the input video or audio file
    :param output_file: Path to save the output caption file
    """
    import datetime

    def ssa_timestamp(seconds):
        """
        Convert seconds to SSA timestamp format (H:MM:SS.CS).
        :param seconds: Timestamp in seconds.
        :return: Formatted SSA timestamp.
        """
        millis = int((seconds % 1) * 100)  # SSA uses centiseconds
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02}:{seconds:02}.{millis:02}"

    print("Loading WhisperX model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("large", device=device, compute_type="float32")

    print(f"Transcribing file: {input_file}")
    transcription = model.transcribe(input_file, verbose=True)

    print("Aligning transcription...")
    segments = transcription.get("segments", [])
    audio_path = transcription.get("audio", None)
    temp_audio_path = "temp_audio.wav"

    if audio_path is None:
        print("Extracting audio from input file for alignment...")
        try:
            clip = AudioFileClip(input_file)
            clip.write_audiofile(temp_audio_path)
            clip.close()
            audio_path = temp_audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            exit(1)

    print("Loading alignment model and metadata...")
    ssl._create_default_https_context = ssl._create_unverified_context
    alignment_model, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)

    alignment = whisperx.align(
        transcript=segments,
        model=alignment_model,
        align_model_metadata=metadata,
        audio=audio_path,
        device=device,
        print_progress=True,
    )

    print("Performing speaker diarization...")
    diarization = diarize_with_progress(audio_path)

    print("Merging transcription with speaker diarization...")
    diarized_segments = merge_segments_with_speakers(alignment["segments"], diarization)

    print(f"Saving output to: {output_file}")
    with open(output_file, "w") as f:
        # Write SSA header
        f.write("[Script Info]\n")
        f.write("Title: Transcription with Diarization\n")
        f.write("ScriptType: v4.00+\n")
        f.write("Collisions: Normal\n")
        f.write("\n[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n")
        f.write("\n[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        # Write each segment
        for segment in diarized_segments:
            start = ssa_timestamp(segment["start"])
            end = ssa_timestamp(segment["end"])
            speaker = segment["speaker"]
            words = segment["words"]

            # Generate SSA dialogue with word-level timing
            dialogue_text = ""
            for word in words:
                word_start = ssa_timestamp(word["start"])
                dialogue_text += f"{{\\k{int((word['end'] - word['start']) * 100)}}}{word['text']} "

            # Add a dialogue event for each segment
            f.write(f"Dialogue: 0,{start},{end},Default,{speaker},0,0,0,,{dialogue_text.strip()}\n")

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    print("Transcription and diarization complete.")

def format_timestamp(seconds):
    """
    Format a timestamp in seconds to VTT format (HH:MM:SS.mmm).

    :param seconds: Timestamp in seconds
    :return: Formatted timestamp string
    """
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions with speaker diarization using WhisperX.")
    parser.add_argument("input_file", type=str, help="Path to the input video or audio file or YouTube URL")
    parser.add_argument("output_file", type=str, nargs="?", default="output.vtt", help="Path to save the output caption file (default: output.vtt)")

    args = parser.parse_args()

    input_file = args.input_file

    if input_file.startswith("http://") or input_file.startswith("https://"):
        # Handle YouTube URL
        input_file = download_audio_from_youtube(input_file)
    elif not os.path.isfile(input_file):
        print(f"Error: File not found: {input_file}")
        exit(1)

    transcribe_with_diarization(input_file, args.output_file)

    # Clean up temporary file if downloaded from YouTube
    if args.input_file.startswith("http://") or args.input_file.startswith("https://"):
        os.remove(input_file)
