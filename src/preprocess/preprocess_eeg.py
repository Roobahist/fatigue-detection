from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import mne
import numpy as np

from Zh.src.utils.common import update_file_name
from Zh.src.utils.file_folder import copy_directory_structure, pick_selected_file_paths
from Zh.src.utils.serialized_parallel import process

NotchType = Union[None, float, int, Sequence[Union[float, int]]]


def _as_list_notch_freq(notch_freq: NotchType) -> list[float]:
    """
    Normalize notch_freq to a list of floats.
    - None -> [50.0, 100.0] for DROZY (Belgium mains + first harmonic)
    - scalar -> [scalar]
    - sequence -> list(sequence)
    """
    if notch_freq is None:
        return [50.0, 100.0]
    if isinstance(notch_freq, (int, float)):
        return [float(notch_freq)]
    return [float(x) for x in notch_freq]


def _infer_channel_groups(raw: mne.io.BaseRaw) -> dict[str, list[str]]:
    """
    Infer EEG and auxiliary channels from DROZY channel names.

    EEG: Fz, Cz, C3, C4, Pz, Oz (Oz may be missing)
    EOG: EOG-V, EOG-H
    EMG: EMG
    ECG: ECG
    Other: Cam-Sync, PVT
    """
    ch_set = set(raw.ch_names)

    eeg_candidates = ["Fz", "Cz", "C3", "C4", "Pz", "Oz"]
    eog_candidates = ["EOG-V", "EOG-H"]
    emg_candidates = ["EMG"]
    ecg_candidates = ["ECG"]

    eeg = [ch for ch in eeg_candidates if ch in ch_set]
    eog = [ch for ch in eog_candidates if ch in ch_set]
    emg = [ch for ch in emg_candidates if ch in ch_set]
    ecg = [ch for ch in ecg_candidates if ch in ch_set]

    return {"eeg": eeg, "eog": eog, "emg": emg, "ecg": ecg}


def _robust_z(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return 0.6744897501960817 * (x - med) / mad


def _detect_bad_eeg_channels_auto(
    raw: mne.io.BaseRaw,
    *,
    z_thresh_var: float = 5.0,
    z_thresh_hf: float = 5.0,
    flat_std_thresh: float = 1e-12,
) -> list[str]:
    """
    Dependency free bad channel detector for EEG.

    Flags channels if:
    - flat (std too small)
    - variance is a strong outlier (robust z on log variance)
    - high frequency noise is a strong outlier (robust z on log HF power)

    This is a fallback when RANSAC is not available.
    """
    raw_eeg = raw.copy().pick_types(eeg=True)
    data = raw_eeg.get_data()  # shape: (n_ch, n_times)
    ch_names = raw_eeg.ch_names
    sfreq = float(raw_eeg.info["sfreq"])

    if data.size == 0:
        return []

    # Flat channels
    std = np.std(data, axis=1)
    flat_mask = std < flat_std_thresh

    # Variance outliers
    var = np.var(data, axis=1)
    z_var = _robust_z(np.log(var + 1e-30))
    var_mask = np.abs(z_var) > z_thresh_var

    # High frequency noise outliers (use 30-45 Hz band, assuming bandpass_high >= 45)
    # If sfreq too low, skip HF detection.
    hf_mask = np.zeros(len(ch_names), dtype=bool)
    if sfreq >= 2 * 45.0:
        # Use Welch PSD on each channel
        nperseg = int(min(data.shape[1], max(256, sfreq * 2)))
        freqs, psd = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=1.0,
            fmax=min(45.0, sfreq / 2.0 - 1e-6),
            n_per_seg=nperseg,
            average="mean",
            verbose=False,
        )
        hf_band = (freqs >= 30.0) & (freqs <= min(45.0, freqs.max()))
        lf_band = (freqs >= 1.0) & (freqs <= 15.0)

        if np.any(hf_band) and np.any(lf_band):
            hf_power = np.mean(psd[:, hf_band], axis=1) + 1e-30
            lf_power = np.mean(psd[:, lf_band], axis=1) + 1e-30
            ratio = hf_power / lf_power
            z_hf = _robust_z(np.log(ratio))
            hf_mask = np.abs(z_hf) > z_thresh_hf

    bad_mask = flat_mask | var_mask | hf_mask
    return [ch for ch, is_bad in zip(ch_names, bad_mask) if bool(is_bad)]


def _detect_bad_channels(
    raw: mne.io.BaseRaw,
    *,
    prefer_ransac: bool = True,
) -> tuple[list[str], str]:
    """
    Auto strategy:
    - If autoreject is available and prefer_ransac True, run RANSAC
    - Otherwise use robust fallback detector
    Returns: (bads, method_name)
    """
    if prefer_ransac:
        try:
            from autoreject import Ransac

            raw_eeg = raw.copy().pick_types(eeg=True)
            # RANSAC expects epochs. Use short fixed-length epochs for stability.
            epochs = mne.make_fixed_length_epochs(
                raw_eeg, duration=2.0, preload=True, verbose=False
            )
            r = Ransac(verbose=False, n_jobs=1)
            r.fit(epochs)
            return list(r.bad_chs_), "autoreject_ransac"
        except Exception:
            print(
                "Autoreject RANSAC not available or failed, falling back to robust stats detector"
            )
            pass

    return _detect_bad_eeg_channels_auto(raw), "robust_stats_fallback"


def preprocess_eeg_recording(
    edf_file_path: str,
    output_file_path: str,
    # Filtering parameters
    bandpass_low: float = 0.5,
    bandpass_high: float = 45.0,
    notch_freq: NotchType = None,
    # ICA parameters
    n_ica_components: Optional[int] = None,
    ica_random_state: int = 42,
    use_eog_channels: bool = True,
    use_ecg_channels: bool = True,
    use_emg_channels: bool = False,
    eog_threshold: float = 3.0,
    ecg_threshold: float = 3.0,
    muscle_threshold: float = 0.8,
    # Resampling
    resample_freq: Optional[float] = None,
    # Reference
    reference: str = "original",
    # Bad channel detection
    detect_bad_channels: bool = True,
    # Visualization
    plot_steps: bool = False,
):
    """
    EEG preprocessing pipeline for DROZY.

    Defaults tuned for DROZY:
    - bandpass: 0.5-45 Hz (safer than 50 for muscle leakage)
    - notch: [50, 100] Hz by default (Belgium mains + 1st harmonic)
    - reference: keep original by default

    Bad channels:
    - Prefer autoreject RANSAC if available, otherwise robust fallback detector.
    """

    report = {}
    print(f"Loading file: {edf_file_path}")

    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    report["original_channels"] = raw.ch_names.copy()
    report["original_sfreq"] = raw.info["sfreq"]
    report["original_duration"] = raw.times[-1]

    groups = _infer_channel_groups(raw)
    eeg_channels = groups["eeg"]
    eog_channels = groups["eog"]
    emg_channels = groups["emg"]
    ecg_channels = groups["ecg"]

    print("Found channels:")
    print(f"  EEG: {eeg_channels}")
    print(f"  EOG: {eog_channels}")
    print(f"  ECG: {ecg_channels}")
    print(f"  EMG: {emg_channels}")

    report["eeg_channels"] = eeg_channels
    report["eog_channels"] = eog_channels
    report["ecg_channels"] = ecg_channels
    report["emg_channels"] = emg_channels

    if not eeg_channels:
        raise RuntimeError(
            "No EEG channels detected. Expected DROZY-style EEG channels: "
            "Fz, Cz, C3, C4, Pz, Oz (Oz may be missing)."
        )

    channels_to_keep = list(eeg_channels)
    if use_eog_channels and eog_channels:
        channels_to_keep.extend(eog_channels)
    if use_ecg_channels and ecg_channels:
        channels_to_keep.extend(ecg_channels)
    if use_emg_channels and emg_channels:
        channels_to_keep.extend(emg_channels)

    raw.pick_channels(channels_to_keep)

    channel_types = {}
    for ch in eeg_channels:
        channel_types[ch] = "eeg"
    for ch in eog_channels:
        channel_types[ch] = "eog"
    for ch in ecg_channels:
        channel_types[ch] = "ecg"
    for ch in emg_channels:
        channel_types[ch] = "emg"
    raw.set_channel_types(channel_types)

    print(f"Selected channels: {raw.ch_names}")
    report["selected_channels"] = raw.ch_names.copy()
    report["n_eeg_channels"] = len(eeg_channels)
    report["n_aux_channels"] = len(raw.ch_names) - len(eeg_channels)

    # Montage (optional)
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")
        print("Montage set successfully")
        report["montage_set"] = True
    except Exception:
        print("Warning: Could not set standard montage")
        report["montage_set"] = False

    # Reference
    if reference != "original":
        raw.set_eeg_reference(reference, projection=False)
        print(f"Re-referenced to {reference}")
        report["reference"] = f"-> {reference}"
    else:
        print("Keeping original reference (as stored in EDF)")
        report["reference"] = "original (as stored)"

    if plot_steps:
        raw.plot(
            duration=10,
            n_channels=len(raw.ch_names),
            title="After re-referencing",
            scalings="auto",
        )
        plt.show()

    # Bandpass filter
    raw.filter(
        l_freq=bandpass_low, h_freq=bandpass_high, fir_design="firwin", verbose=False
    )
    print(f"Applied bandpass filter: {bandpass_low}-{bandpass_high} Hz")
    report["bandpass_filter"] = (bandpass_low, bandpass_high)

    if plot_steps:
        raw.plot(
            duration=10,
            n_channels=len(raw.ch_names),
            title="After bandpass filtering",
            scalings="auto",
        )
        plt.show()

    # Notch filter (DROZY Belgium defaults)
    notch_list = _as_list_notch_freq(notch_freq)
    raw.notch_filter(freqs=notch_list, verbose=False)
    print(f"Applied notch filter at {notch_list} Hz")
    report["notch_filter"] = notch_list

    # Bad channel detection and interpolation
    if detect_bad_channels:
        try:
            bads, method = _detect_bad_channels(raw, prefer_ransac=True)
            report["bad_channel_method"] = method

            if bads:
                raw.info["bads"] = bads
                print(f"Bad channels detected ({method}): {bads}")
                raw.interpolate_bads(reset_bads=True, verbose=False)
                print("Bad channels interpolated")
                report["bad_channels"] = bads
            else:
                print(f"No bad channels detected ({method})")
                report["bad_channels"] = []
        except Exception as e:
            print(f"Bad channel detection failed: {e}")
            report["bad_channels"] = "detection_failed"
            report["bad_channel_method"] = "failed"
    else:
        report["bad_channels"] = "not_performed"
        report["bad_channel_method"] = "not_performed"

    # ICA
    print("Running ICA for artifact removal...")

    n_eeg = len(eeg_channels)
    default_n_ica = max(2, min(n_eeg - 1, 25))
    n_components = (
        default_n_ica
        if n_ica_components is None
        else min(n_eeg - 1, int(n_ica_components))
    )
    report["ica_n_components"] = n_components

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=ica_random_state,
        max_iter="auto",
        verbose=False,
    )

    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
    ica.fit(raw_ica.copy().pick_types(eeg=True), verbose=False)

    excluded_components: list[int] = []

    if use_eog_channels and eog_channels:
        try:
            eog_indices, _ = ica.find_bads_eog(raw_ica, ch_name=eog_channels, threshold=eog_threshold, verbose=False)
            excluded_components.extend(eog_indices)
            print(f"ICA components excluded (EOG): {eog_indices}")
            report["ica_eog_components"] = eog_indices
        except Exception as e:
            print(f"Could not perform EOG detection: {e}")
            report["ica_eog_components"] = "detection_failed"
    else:
        report["ica_eog_components"] = "not_performed"

    if use_ecg_channels and ecg_channels:
        try:
            ecg_indices, _ = ica.find_bads_ecg(raw_ica, ch_name=ecg_channels[0], threshold=ecg_threshold, verbose=False)
            excluded_components.extend(ecg_indices)
            print(f"ICA components excluded (ECG): {ecg_indices}")
            report["ica_ecg_components"] = ecg_indices
        except Exception as e:
            print(f"Could not perform ECG detection: {e}")
            report["ica_ecg_components"] = "detection_failed"
    else:
        report["ica_ecg_components"] = "not_performed"

    # Muscle IC detection does not require EMG channels
    if use_emg_channels:
        try:
            muscle_indices, _ = ica.find_bads_muscle(
                raw_ica, threshold=muscle_threshold, verbose=False
            )
            excluded_components.extend(muscle_indices)
            print(f"ICA components excluded (Muscle): {muscle_indices}")
            report["ica_muscle_components"] = muscle_indices
        except Exception as e:
            print(f"Could not perform muscle detection: {e}")
            report["ica_muscle_components"] = "detection_failed"
    else:
        report["ica_muscle_components"] = "not_performed"

    ica.exclude = sorted(set(excluded_components))
    report["ica_excluded_components"] = ica.exclude

    raw_clean = raw.copy()
    ica.apply(raw_clean, picks="eeg", verbose=False)
    print(f"ICA applied to EEG channels only. Excluded {len(ica.exclude)} components")

    if plot_steps:
        ica.plot_sources(raw, show_scrollbars=False)
        plt.show()
        raw.plot(
            duration=10,
            n_channels=len(eeg_channels),
            title="Before ICA",
            scalings="auto",
        )
        raw_clean.plot(
            duration=10,
            n_channels=len(eeg_channels),
            title="After ICA",
            scalings="auto",
        )
        plt.show()

    raw_preprocessed = raw_clean.copy()
    report["epoch_rejection"] = "not_performed"

    # Resample
    if resample_freq is not None:
        raw_preprocessed.resample(resample_freq, verbose=False)
        print(f"Resampled to {resample_freq} Hz")
        report["resampled_freq"] = resample_freq
    else:
        report["resampled_freq"] = "not_performed"

    print("\n" + "=" * 50)
    print("Preprocessing completed!")
    print(f"Final sampling rate: {raw_preprocessed.info['sfreq']} Hz")
    print(f"Final number of channels: {len(raw_preprocessed.ch_names)}")
    print(f"Final duration: {raw_preprocessed.times[-1]:.2f} seconds")
    print("=" * 50 + "\n")

    report["final_sfreq"] = raw_preprocessed.info["sfreq"]
    report["final_channels"] = raw_preprocessed.ch_names
    report["final_duration"] = raw_preprocessed.times[-1]

    raw_preprocessed.save(output_file_path, overwrite=True, verbose=False)
    print(f"\nSaved preprocessed data to: {output_file_path}")

    return raw_preprocessed, report


def preprocess_eeg_recordings(
    input_folder: str,
    relative_selected_paths: Iterable[str],
    output_folder: str = "",
    file_indices_spec: str = "",
    bandpass_low: float = 0.5,
    bandpass_high: float = 45.0,
    notch_freq: NotchType = None,
    n_ica_components: Optional[int] = None,
    ica_random_state: int = 42,
    use_eog_channels: bool = True,
    use_ecg_channels: bool = True,
    use_emg_channels: bool = False,
    eog_threshold: float = 3.0,
    ecg_threshold: float = 3.0,
    muscle_threshold: float = 0.8,
    resample_freq: Optional[float] = None,
    reference: str = "original",
    detect_bad_channels: bool = True,
    plot_steps: bool = False,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    maxtasksperchild: Optional[int] = None,
    use_tqdm: bool = True,
):
    if not output_folder or output_folder == input_folder:
        output_folder = input_folder
    else:
        copy_directory_structure(input_folder, output_folder)

    file_index_to_path = pick_selected_file_paths(
        input_folder=input_folder,
        depth=1,
        EXT_LIST=[".edf"],
        relative_selected_paths=relative_selected_paths,
        file_indices_spec=file_indices_spec,
        sort_and_dedupe=True,
    )

    if parallel and plot_steps:
        logging.warning(
            "plot_steps=True with parallel=True is not supported. Disabling plotting."
        )
        plot_steps = False

    args_list = [
        {
            "edf_file_path": file_path,
            "output_file_path": update_file_name(
                file_path=file_path,
                input_folder=input_folder,
                output_folder=output_folder,
            ),
            "bandpass_low": bandpass_low,
            "bandpass_high": bandpass_high,
            "notch_freq": notch_freq,
            "n_ica_components": n_ica_components,
            "ica_random_state": ica_random_state,
            "use_eog_channels": use_eog_channels,
            "use_ecg_channels": use_ecg_channels,
            "use_emg_channels": use_emg_channels,
            "eog_threshold": eog_threshold,
            "ecg_threshold": ecg_threshold,
            "muscle_threshold": muscle_threshold,
            "resample_freq": resample_freq,
            "reference": reference,
            "detect_bad_channels": detect_bad_channels,
            "plot_steps": plot_steps,
        }
        for _file_index, file_path in file_index_to_path.items()
    ]

    process(
        worker_function=preprocess_eeg_recording,
        worker_params_list=args_list,
        handler_function=None,
        parallel=parallel,
        max_workers=max_workers,
        maxtasksperchild=maxtasksperchild,
        use_tqdm=use_tqdm,
        auto_slice=True,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="preprocess EEG EDF files")

    parser.add_argument(
        "input_folder", type=str, help="Path to the input folder containing EDF files."
    )
    parser.add_argument(
        "relative_selected_paths",
        nargs="*",
        help="Relative EDF paths to process. If empty, all EDFs in input_folder are processed.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="",
        help="Output folder. If not specified, input_folder is used.",
    )
    parser.add_argument(
        "--file_indices_spec",
        type=str,
        default="",
        help="Comma-separated indices/ranges, example: '0-9,20,35-'",
    )

    parser.add_argument("--bandpass_low", type=float, default=0.5)
    parser.add_argument("--bandpass_high", type=float, default=45.0)

    parser.add_argument(
        "--notch_freq",
        type=float,
        nargs="+",
        default=None,
        help="One or more notch frequencies. If omitted, defaults to 50 and 100 for DROZY.",
    )

    parser.add_argument(
        "--n_ica_components",
        type=int,
        default=None,
        help="Number of ICA components. If omitted, uses min(n_eeg-1, 25).",
    )
    parser.add_argument("--ica_random_state", type=int, default=42)

    parser.add_argument(
        "--use_eog_channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EOG channels (EOG-V, EOG-H) for ICA artifact detection.",
    )
    parser.add_argument(
        "--use_ecg_channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ECG channel for ICA artifact detection.",
    )
    parser.add_argument(
        "--use_emg_channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable muscle IC detection (does not require EMG channel).",
    )
    parser.add_argument("--eog_threshold", type=float, default=3.0)
    parser.add_argument("--ecg_threshold", type=float, default=3.0)
    parser.add_argument("--muscle_threshold", type=float, default=0.8)

    parser.add_argument("--resample_freq", type=float, default=None)
    parser.add_argument("--reference", type=str, default="original")

    parser.add_argument(
        "--detect_bad_channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect and interpolate bad EEG channels. Uses autoreject RANSAC if available, else robust fallback.",
    )
    parser.add_argument(
        "--plot_steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot intermediate steps. Not supported with parallel=True.",
    )

    parser.add_argument(
        "--parallel", action="store_true", help="Process files in parallel."
    )
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--maxtasksperchild", type=int, default=None)
    parser.add_argument(
        "--no_tqdm",
        action="store_false",
        dest="use_tqdm",
        default=True,
        help="Disable tqdm progress bars.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.info("Starting ...")

    preprocess_eeg_recordings(
        input_folder=args.input_folder,
        relative_selected_paths=args.relative_selected_paths,
        output_folder=args.output_folder,
        file_indices_spec=args.file_indices_spec,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        notch_freq=args.notch_freq,
        n_ica_components=args.n_ica_components,
        ica_random_state=args.ica_random_state,
        use_eog_channels=args.use_eog_channels,
        use_ecg_channels=args.use_ecg_channels,
        use_emg_channels=args.use_emg_channels,
        eog_threshold=args.eog_threshold,
        ecg_threshold=args.ecg_threshold,
        muscle_threshold=args.muscle_threshold,
        resample_freq=args.resample_freq,
        reference=args.reference,
        detect_bad_channels=args.detect_bad_channels,
        plot_steps=args.plot_steps,
        parallel=args.parallel,
        max_workers=args.max_workers,
        maxtasksperchild=args.maxtasksperchild,
        use_tqdm=args.use_tqdm,
    )
