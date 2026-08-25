"""Denoise a signal in the frequency domain and build windowed spectra."""

import numpy as np


TOLERANCES = {
    "filtered_head": (1e-12, 1e-12),
    "spectrogram_energy": (1e-12, 1e-12),
}


def results():
    sample_rate = 64.0
    count = 128
    time = np.arange(count) / sample_rate
    clean = np.sin(2.0 * np.pi * 5.0 * time)
    noise = np.random.default_rng(20260825).normal(0.0, 0.35, count)
    noisy = clean + noise

    frequencies = np.fft.rfftfreq(count, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(noisy)
    filtered = np.fft.irfft(np.where(frequencies <= 8.0, spectrum, 0.0), n=count)

    windows = np.lib.stride_tricks.sliding_window_view(noisy, 32)[::16]
    tapered = windows * np.hanning(32)
    spectrogram = np.abs(np.fft.rfft(tapered, axis=1)) ** 2
    peak_bins = np.argmax(spectrogram[:, 1:], axis=1) + 1
    window_frequencies = np.fft.rfftfreq(32, d=1.0 / sample_rate)
    return {
        "peak_frequencies": window_frequencies[peak_bins],
        "filtered_head": filtered[:8],
        "spectrogram_energy": spectrogram[:, 1:].sum(axis=1),
    }


def main():
    out = results()
    print("window peak frequencies:", out["peak_frequencies"])
    print("filtered head:", np.round(out["filtered_head"], 6))
    print("window energies:", np.round(out["spectrogram_energy"], 3))
    assert np.array_equal(out["peak_frequencies"], [4.0, 6.0, 4.0, 4.0, 6.0, 4.0, 6.0])
    assert np.allclose(out["filtered_head"][:3], [0.14153594, 0.63147300, 0.93180457], rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    main()
