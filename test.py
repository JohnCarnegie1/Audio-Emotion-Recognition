import os
import wave
import struct
from pathlib import Path

def load_wav_files(folder_path):
    """
    Load all WAV files from a folder.
    Returns a list of dicts with filename, audio data, and metadata.
    """
    folder = Path(folder_path)
    results = []

    wav_files = list(folder.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        return results

    for wav_path in sorted(wav_files):
        try:
            with wave.open(str(wav_path), 'rb') as wf:
                n_channels   = wf.getnchannels()
                sample_width = wf.getsampwidth()   # bytes per sample
                frame_rate   = wf.getframerate()   # Hz
                n_frames     = wf.getnframes()
                duration     = n_frames / frame_rate

                raw_bytes = wf.readframes(n_frames)

            # Unpack raw bytes into integer samples
            fmt = {1: 'B', 2: 'h', 4: 'i'}.get(sample_width)
            if fmt is None:
                raise ValueError(f"Unsupported sample width: {sample_width} bytes")
            total_samples = n_frames * n_channels
            samples = list(struct.unpack(f"{total_samples}{fmt}", raw_bytes))

            details = list(wav_path.name)
            gender = details[0]
            emotion = "".join(details[2:5])
            language = "".join(details[9:11])

            # print(gender)
            
            # with open("label.txt", "a") as file:
            #     if emotion == "ans":
            #         file.write(wav_path.name + ",fear\n")
            #     elif emotion == "dis":
            #         file.write(wav_path.name + ",disgust\n")
            #     elif emotion == "gio":
            #         file.write(wav_path.name + ",happiness\n")
            #     elif emotion == "rab":
            #         file.write(wav_path.name + ",anger\n")
            #     elif emotion == "tri":
            #         file.write(wav_path.name + ",sadness\n")
                
                

            results.append({
                "filename":     wav_path.name,
                "path":         str(wav_path),
                "gender":       gender,
                "emotion":      emotion,
                "language":     language,
                "channels":     n_channels,
                "sample_width": sample_width,
                "frame_rate":   frame_rate,
                "n_frames":     n_frames,
                "duration_sec": round(duration, 3),
                "samples":      samples,
            })
            print(f"  Loaded: {wav_path.name} | {frame_rate} Hz | "
                  f"{n_channels}ch | {duration:.2f}s")

        except Exception as e:
            print(f"  Error loading {wav_path.name}: {e}")

    print(f"\nLoaded {len(results)}/{len(wav_files)} WAV files from '{folder_path}'")
    return results


# ── Example usage ────────────────────────────────────────
if __name__ == "__main__":
    folder = "/Users/john6/OneDrive - Gonzaga University/6. MSDS/Research - Emotion Detection/wav_corpus/"
    audio_files = load_wav_files(folder)

    # for audio in audio_files:
    #     print(f"\n{audio['filename']}")
    #     print(f"  Sample rate : {audio['frame_rate']} Hz")
    #     print(f"  Channels    : {audio['channels']}")
    #     print(f"  Duration    : {audio['duration_sec']} seconds")
    #     print(f"  Total samples: {len(audio['samples'])}")