import os
import gc
import warnings
import librosa
import numpy as np
import scipy.signal as signal
import pyloudnorm as pyln
import arabic_reshaper
from bidi.algorithm import get_display
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime

# -- Warning Settings --
warnings.filterwarnings("ignore")

# -- Helper functions for Persian text --
def reshape_text(text: str) -> str:
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

# -- Load audio file --
def load_audio(file_path):
    try:
        data, sr = librosa.load(file_path, sr=None, mono=True)
        return data, sr
    except Exception as e:
        # Log error inside the function for tracking in parallel processing
        print(f"[Error] load_audio '{os.path.basename(file_path)}': {e}")
        return None, None

# -- Calculate duration (minutes) --
def calculate_duration(data, sr):
    if data is None or sr in (None, 0):
        return 0.0
    seconds = len(data) / sr
    return seconds / 60.0

# -- Calculate LUFS --
def calculate_lufs(data, sr):
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)
        return float(loudness) if np.isfinite(loudness) else 0.0
    except Exception:
        return 0.0

# -- Calculate approximate DR --
def calculate_dr(data, sr):
    try:
        if data is None or sr in (None, 0):
            return 0.0
        segment_length = 3 * sr
        rms_values = []
        for i in range(0, len(data), segment_length):
            seg = data[i:i+segment_length]
            if seg.size == 0:
                continue
            rms = np.sqrt(np.mean(seg**2))
            if rms > 0:
                rms_db = 20 * np.log10(rms)
                if np.isfinite(rms_db):
                    rms_values.append(rms_db)
        if len(rms_values) > 1:
            dr = np.max(rms_values) - np.min(rms_values)
            return float(dr) if np.isfinite(dr) else 0.0
        return 0.0
    except Exception:
        return 0.0

# -- Calculate RMS (in dB) --
def calculate_rms(data):
    try:
        if data is None:
            return 0.0
        rms = np.sqrt(np.mean(data**2))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
            return float(rms_db) if np.isfinite(rms_db) else 0.0
        return 0.0
    except Exception:
        return 0.0

# -- Calculate True Peak (approximate, dB) --
def calculate_true_peak(data, sr):
    try:
        if data is None:
            return 0.0
        # Simple upsampling for True Peak approximation
        resampled = signal.resample(data, len(data) * 2)
        true_peak = np.max(np.abs(resampled))
        if true_peak > 0:
            tp_db = 20 * np.log10(true_peak)
            return float(tp_db) if np.isfinite(tp_db) else 0.0
        return 0.0
    except Exception:
        return 0.0

# -- Calculate 1/f noise slope (approximate) --
def calculate_one_over_f_noise(data, sr):
    try:
        if data is None or len(data) < 10:
            return 0.0
        fft = np.fft.rfft(data)
        freq = np.fft.rfftfreq(len(data), 1/sr)
        power = np.abs(fft)**2 / len(data)
        mask = (freq > 1) & (freq < 100)
        if np.sum(mask) < 2:
            return 0.0
        freq_log = np.log10(freq[mask])
        power_log = np.log10(power[mask] + 1e-12)
        slope, _ = np.polyfit(freq_log, power_log, 1)
        return float(slope) if np.isfinite(slope) else 0.0
    except Exception:
        return 0.0

# -- Analyze one audio file (this function runs in child processes) --
def analyze_audio(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    try:
        data, sr = load_audio(file_path)
        if data is None or sr is None:
            raise RuntimeError("Unable to load audio")

        duration = calculate_duration(data, sr)
        lufs = calculate_lufs(data, sr)
        dr = calculate_dr(data, sr)
        rms = calculate_rms(data)
        true_peak = calculate_true_peak(data, sr)
        one_f_noise = calculate_one_over_f_noise(data, sr)

        # Attempt to free memory in child process before returning
        try:
            del data
            gc.collect()
        except Exception:
            pass

        return {
            'song': file_name,
            'duration': duration,
            'lufs': lufs,
            'dr': dr,
            'rms': rms,
            'true_peak': true_peak,
            'one_f_noise': one_f_noise
        }
    except Exception as e:
        print(f"[Error] processing '{file_name}': {e}")
        return {
            'song': file_name,
            'duration': 0.0,
            'lufs': 0.0,
            'dr': 0.0,
            'rms': 0.0,
            'true_peak': 0.0,
            'one_f_noise': 0.0
        }

# -- Process folder with pool.imap and tqdm --
def analyze_folder(folder_path):
    audio_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.mp3', '.wav'))])
    if not audio_files:
        print(reshape_text("Error: No audio files (.mp3 or .wav) found."))
        return None

    num_processes = min(cpu_count(), max(1, len(audio_files)))
    print(reshape_text(f"\nNumber of processes: {num_processes} (based on {cpu_count()} CPU cores)"))

    results = {
        'songs': [],
        'durations': [],
        'lufs': [],
        'dr': [],
        'rms': [],
        'true_peak': [],
        'one_f_noise': []
    }

    analyze_partial = partial(analyze_audio, folder_path=folder_path)

    # With imap (order preserved) and tqdm for live progress bar
    with Pool(processes=num_processes) as pool:
        try:
            for res in tqdm(pool.imap(analyze_partial, audio_files), total=len(audio_files),
                            desc=reshape_text("Analyzing songs"), unit=reshape_text("song"),
                            ncols=90, leave=True):
                results['songs'].append(res['song'])
                results['durations'].append(res['duration'])
                results['lufs'].append(res['lufs'])
                results['dr'].append(res['dr'])
                results['rms'].append(res['rms'])
                results['true_peak'].append(res['true_peak'])
                results['one_f_noise'].append(res['one_f_noise'])
        except KeyboardInterrupt:
            print(reshape_text("\nProcessing stopped by user!"))
            pool.terminate()
            pool.join()
            raise

    return results

# -- Display results in terminal and save to results.txt --
def display_and_save_results(results, out_txt_path):
    if not results or not results['songs']:
        print(reshape_text("No results to display."))
        return

    headers = [reshape_text(h) for h in ['Song Name', 'Duration (minutes)', 'LUFS', 'DR (dB)', 'RMS (dB)', 'True Peak (dB)', '1/f Noise Slope']]
    table = []
    for i in range(len(results['songs'])):
        table.append([
            results['songs'][i],
            f"{results['durations'][i]:.2f}",
            f"{results['lufs'][i]:.2f}",
            f"{results['dr'][i]:.2f}",
            f"{results['rms'][i]:.2f}",
            f"{results['true_peak'][i]:.2f}",
            f"{results['one_f_noise'][i]:.2f}"
        ])

    output = tabulate(table, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center")
    print("\n" + output + "\n")

    # Save to text file (UTF-8)
    try:
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(f"Audio Analysis Report - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(output)
        print(reshape_text(f"Results saved to file: {out_txt_path}"))
    except Exception as e:
        print(f"[Error] saving results to '{out_txt_path}': {e}")

# --------------------- Entry point ---------------------
def main():
    freeze_support()
    folder_path = input(reshape_text("Enter the path to the songs folder: ")).strip()
    if not os.path.isdir(folder_path):
        print(reshape_text("Error: The entered path is invalid!"))
        return

    print(reshape_text("\nList of found songs:"))
    audio_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.mp3', '.wav'))])
    for idx, name in enumerate(audio_files, 1):
        display_name = name if len(name) <= 60 else name[:57] + "..."
        print(f"{idx}. {display_name}")

    results = analyze_folder(folder_path)
    if results is None:
        return

    out_txt = os.path.join(folder_path, "results.txt")
    display_and_save_results(results, out_txt)

if __name__ == "__main__":
    main()