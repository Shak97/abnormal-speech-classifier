import os
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pathlib import Path

def split_audio_with_precision(source_path, output_root, model_name="medium"):
    print(f"Loading Whisper model ({model_name})...")
    model = whisper.load_model(model_name)

    extensions = {".mp3", ".wav", ".m4a", ".flac", ".aac"}
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_path)
                current_output_dir = os.path.join(output_root, relative_path, Path(file).stem)
                os.makedirs(current_output_dir, exist_ok=True)

                print(f"\n[Processing] {input_file_path}")
                process_with_trimming(input_file_path, current_output_dir, model)

def process_with_trimming(file_path, output_dir, model):
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        print(f"   [Error] Load failed: {e}")
        return

    # 1. Get Whisper segments (The 'Draft' timestamps)
    result = model.transcribe(file_path, language="ko")
    segments = result['segments']

    saved_count = 0

    for i, seg in enumerate(segments):
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        text = seg['text'].strip()

        rough_chunk = audio[max(0, start_ms - 300) : min(len(audio), end_ms + 300)]

        nonsilent_ranges = detect_nonsilent(
            rough_chunk, 
            min_silence_len=300, 
            silence_thresh=-45
        )

        if not nonsilent_ranges:
            print(f"   [Skipped] Segment {i} contained no actual sound.")
            continue
        actual_start = nonsilent_ranges[0][0]
        actual_end = nonsilent_ranges[-1][1]
        

        final_chunk = rough_chunk[max(0, actual_start - 100) : min(len(rough_chunk), actual_end + 100)]

        clean_text = "".join(filter(str.isalnum, text))[:30]
        if not clean_text: continue

        output_path = os.path.join(output_dir, f"{i:04d}_{clean_text}.wav")
        final_chunk.export(output_path, format="wav", parameters=["-ar", "16000"])
        saved_count += 1

    print(f"   [Done] Saved {saved_count} precise segments.")

if __name__ == "__main__":
    SOURCE_DIRECTORY = r"013.구음장애 음성인식 데이터\01.데이터\1.Training\원천데이터\TS02_언어청각장애"
    OUTPUT_DIRECTORY = r"D:\speechdata\speech_abnormal_data\train\2"
    
    split_audio_with_precision(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, model_name="medium")