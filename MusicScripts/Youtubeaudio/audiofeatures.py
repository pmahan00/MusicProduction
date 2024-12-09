import os
import shutil
import json
from pytubefix import YouTube
import subprocess
import librosa
import librosa.display
import numpy as np
from music21 import chord, stream, note
import matplotlib.pyplot as plt

def download_audio(url, start_time=None, end_time=None):
    yt = YouTube(url)
    print(f"Title: {yt.title}")

    # Download audio stream only
    ys = yt.streams.get_audio_only()
    filename = ys.default_filename.replace(' ', '')
    output_dir = 'ytfolder'

    # Create 'ytfolder' directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Download the audio file
    output_path = ys.download()
    print(f"Downloaded audio file: {output_path}")

    # Convert to mp3
    mp3_filename = os.path.splitext(filename)[0] + '.mp3'
    mp3_output_path = os.path.join(output_dir, mp3_filename)
    subprocess.run(['ffmpeg', '-i', output_path, '-acodec', 'mp3', mp3_output_path])
    print(f"Converted to mp3: {mp3_output_path}")

    # If start_time and end_time are provided, extract the specified clip
    clip_output_path = mp3_output_path
    if start_time and end_time:
        clip_output_path = os.path.join(output_dir, f'clip_{mp3_filename}')
        subprocess.run(['ffmpeg', '-ss', start_time, '-to', end_time, '-i', mp3_output_path, '-acodec', 'mp3', clip_output_path])
        print(f"Extracted clip: {clip_output_path}")

    os.remove(output_path)  # Clean up the original download
    return clip_output_path

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    results = {}

    # Acoustic Features
    print("Extracting Acoustic Features...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = np.max(rms) - np.min(rms)

    # Convert numpy arrays to lists or floats
    results['Tempo_BPM'] = float(tempo)
    results['Dynamic_Range'] = float(dynamic_range)
    results['Spectral_Centroid_Mean'] = float(np.mean(spectral_centroids))
    results['Spectral_Bandwidth_Mean'] = float(np.mean(spectral_bandwidth))
    results['MFCCs_Mean'] = mfccs.mean(axis=1).tolist()

    # Plot and save figures
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.7)
    plt.title("Waveform")
    plt.savefig(os.path.join(output_dir, 'waveform.png'))

    plt.subplot(4, 1, 2)
    plt.semilogy(spectral_centroids, label="Spectral Centroid")
    plt.title("Spectral Centroid")
    plt.savefig(os.path.join(output_dir, 'spectral_centroid.png'))

    plt.subplot(4, 1, 3)
    librosa.display.specshow(mfccs, sr=sr, x_axis="time", cmap="coolwarm")
    plt.colorbar()
    plt.title("MFCCs")
    plt.savefig(os.path.join(output_dir, 'mfccs.png'))

    plt.subplot(4, 1, 4)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", cmap="coolwarm")
    plt.colorbar()
    plt.title("Chroma Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chroma_features.png'))
    plt.close()

    return results

def extract_harmony(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y_harmonic, _ = librosa.effects.hpss(y)
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)
    pitch_threshold = 0.1  # Magnitude threshold
    detected_pitches = [librosa.hz_to_midi(p) for p in np.concatenate([pitches.T[i][pitches.T[i] > pitch_threshold] for i in range(len(pitches.T))])]
    midi_pitches = [int(p) for p in detected_pitches if not np.isnan(p)]
    m21_notes = [note.Note(p) for p in midi_pitches]

    s = stream.Stream(m21_notes)
    analyzed_chords = []
    for chord_obj in s.chordify().flatten().getElementsByClass(chord.Chord):  # Updated to use .flatten()
        analyzed_chords.append({
            "Chord": chord_obj.commonName,
            "Root": chord_obj.root().name,
            "Quality": chord_obj.quality
        })

    return analyzed_chords[:10]  # Limit to 10 for simplicity

if __name__ == "__main__":
    # Download and process
    url = input("Enter the YouTube URL: ")
    start_time_input = input("Enter the start time in seconds (or press X to skip): ")
    start_time = start_time_input if start_time_input.lower() != 'x' else None
    end_time_input = input("Enter the end time in seconds (or press X to skip): ")
    end_time = end_time_input if end_time_input.lower() != 'x' else None

    audio_path = download_audio(url, start_time, end_time)

    # Analyze the downloaded audio
    acoustic_results = analyze_audio(audio_path)
    harmony_results = extract_harmony(audio_path)

    # Combine results and save as JSON
    output_dir = 'analysis_results'
    results = {"Acoustic_Features": acoustic_results, "Harmony_Analysis": harmony_results}
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("Analysis complete. Results and plots are saved in the 'analysis_results' directory.")
