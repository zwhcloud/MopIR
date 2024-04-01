import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def moving_average(data, window_size):
    """Smooth data by doing a moving average"""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def vegetation(baseline_lower=0, 
               baseline_upper=0.5, 
               noise_intensity=0.01,
               extra_noise_lower=0.01, 
               extra_noise_upper=0.03, 
               window_size=70,
               index_ranges_to_smooth=[(150, 170), (205, 225), (480, 500), (550, 570)]):
    """
    Generates a spectrum with specific characteristics including baseline adjustment,
    noise, and additional noise in a specified range. The spectrum is then smoothed using
    a moving average and specific ranges are further smoothed using uniform_filter1d.
    
    Parameters:
    - baseline_lower: float, lower limit of baseline adjustment.
    - baseline_upper: float, upper limit of baseline adjustment.
    - noise_intensity: float, intensity of the initial added noise.
    - extra_noise_lower: float, lower limit of additional noise intensity.
    - extra_noise_upper: float, upper limit of additional noise intensity.
    - window_size: int, window size for the moving average smoothing.
    - index_ranges_to_smooth: list of tuples, specifies the start and end indices of ranges to smooth further.
    
    Returns:
    - adjusted_spectrum: ndarray, the adjusted and smoothed spectrum.
    - wavelengths: ndarray, the array of wavelengths corresponding to the spectrum.
    """
    # Parameters
    frequencies = np.array([400, 550, 900, 950, 1100, 1150, 1250, 1350, 1650, 1800, 1950, 2250, 4000])  # cm^-1
    intensities = np.array([0, 0.1, 0.05, -0.01, 0.05, -0.03, 0.01, -0.01, 0.2, 0.15, -0.1, 0.1, 0])
    peak_widths = np.array([50, 70, 20, 50, 50, 100, 100, 50, 100, 50, 200, 200, 1500])

    # Generate wavelengths
    wavelengths = np.linspace(400, 4000, 2000)

    # Main spectrum generation loop
    spectrum = np.zeros_like(wavelengths)
    for f, i, w in zip(frequencies, intensities, peak_widths):
        peak = i * np.exp(-0.5 * ((wavelengths - f) / w)**2)
        spectrum += peak

    # Adjust the baseline using the provided baseline_lower and baseline_upper
    baseline_adjustment = ((wavelengths > 750) & (wavelengths < 1350)) * np.random.uniform(baseline_lower, baseline_upper)
    spectrum += baseline_adjustment

    # Add noise based on the provided noise_intensity
    spectrum += np.random.normal(0, noise_intensity, wavelengths.shape)

    # Increase noise in a specific range based on the provided extra_noise_lower and extra_noise_upper
    additional_noise_range = (wavelengths > 3500) & (wavelengths < 4000)
    additional_noise_intensity = np.random.uniform(extra_noise_lower, extra_noise_upper)
    spectrum[additional_noise_range] += np.random.normal(0, additional_noise_intensity, sum(additional_noise_range))

    # Smooth the spectrum using the provided window_size
    smooth_spectrum = moving_average(spectrum, window_size=window_size)

    # Incorrect the baseline with a varying pattern
    #varying_baseline = 0.1 * np.sin(0.001 * wavelengths + np.random.uniform(baseline_lower, baseline_upper)) + 0.1
    varying_baseline = 0.1 * np.sin(0.001 * wavelengths) + 0.1# np.random.uniform(baseline_lower, baseline_upper)
    adjusted_spectrum = smooth_spectrum + varying_baseline
    #adjusted_spectrum = smooth_spectrum

    # Smooth specific ranges by uniform_filter1d based on the provided index_ranges_to_smooth
    for start_idx, end_idx in index_ranges_to_smooth:
        adjusted_spectrum[start_idx:end_idx] = uniform_filter1d(adjusted_spectrum[start_idx:end_idx], size=20)

    # Return the wavelengths and the adjusted_spectrum
    return adjusted_spectrum, wavelengths
