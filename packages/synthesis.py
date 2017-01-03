import argparse
import json
import numpy as np
import numexpr as ne
from h5store import *
from concurrent.futures import ProcessPoolExecutor as Pool
from sonorus import create_logger, recursive_mapping_update
from acoustics import Signal
from acoustics.generator import pink
from acoustics.signal import OctaveBand

logger = create_logger(__name__)

# Pivot features

def _features_as_function_of_time(noise, tones):
    """Pivot features to obtain them as function of time.

    :param noise: DataFrame with noise components.
    :param tones: DataFrame with tonal components.
    :returns: Three series, one with noise levels as function of time, one with tone frequencies as function of time and one with tone levels as function of time.
    """
    noise_levels = noise.pivot(columns='nominal', values='level', index='datetime')
    tones_frequencies = tones.pivot(columns='harmonic', values='frequency', index='datetime')
    tones_levels = tones.pivot(columns='harmonic', values='level', index='datetime')
    tones_bandwidth = tones.pivot(columns='harmonic', values='bandwidth', index='datetime')

    # Postprocess features
    noise_levels = _features_fill_missing_data(noise_levels)
    tones_frequencies = _features_fill_missing_data(tones_frequencies)
    tones_levels = _features_fill_missing_data(tones_levels)
    tones_bandwidth = _features_fill_missing_data(tones_bandwidth)

    return noise_levels, tones_frequencies, tones_levels, tones_bandwidth


def _features_fill_missing_data(features):
    """Fill missing values in missing data.

    :param features: Features as function of time.
    :returns: Features as function of time with missing data filled.
    """
    return features.replace(-np.inf, np.nan).interpolate().dropna(axis='columns')

# ------------------- Generate emission ---------------------

SETTINGS = {
    'window' : 3.0, # Size of rolling window features.
    'fs' : 44100, # Target sample frequency.
    'noise_fraction': 3,
    }

def _interpolate_and_fill(values, datetimes, fs, window_seconds=None):
    """Interpolate dataframe/series with values at datetime.

    :param values: Values to interpolate.
    :param datetimes: Times for which to determine a value.
    :param fs: Sample frequency.
    :param window_seconds: Rolling median window in seconds.
    """
    out = values.reindex(datetimes, method='nearest', limit=1).interpolate('linear').fillna(method='backfill')

    if window_seconds is not None:
        #out = pd.rolling_mean(out, window=int(fs*window_seconds), min_periods=1, center=True)
        out = out.rolling(window=int(fs*window_seconds), min_periods=1, center=True).mean()
    return out

def _tone(frequency, times, phase):
    pi = np.pi
    return ne.evaluate("sin(2.* pi * frequency * times + phase)")

def _generate_tone(datetimes, fs, frequency, level, bandwidth, window_seconds, phase=0.0):
    """Generate tone with time-variant level and frequency.

    Tones are assigned a random phase.

    :param datetimes: Datetimes for which a sample is calculated.
    :param fs: Sample frequency.
    :param level: Sound pressure level.
    :param frequency: Frequency of the sine.
    :param noise: Noise to filter. Signal class.
    """
    nsamples = len(datetimes)
    #print(level)
    if np.all(frequency) < fs/2.0:
        #bandwidth = _interpolate_and_fill(bandwidth, datetimes, fs, window_seconds).values.T
        bandwidth = bandwidth.mean()
        frequency = frequency.mean()
        times = np.arange(nsamples) / fs
        ###frequency = _interpolate_and_fill(frequency, datetimes, fs, window_seconds).values.T
        level = _interpolate_and_fill(level, datetimes, fs, window_seconds).values.T

        tone = Signal(_tone(frequency, times, phase), fs).calibrate_to(level)
        #if noise is None:
            #noise = Signal(pink(nsamples), fs)
        #noise = noise.bandpass(frequency-bandwidth/2.0, frequency+bandwidth/2.0, zero_phase=True).calibrate_to(level)
        signal = tone #+ noise
        #signal = noise
        return signal.calibrate_to(level)
    else:
        return 0.0

def _generate_noise(datetimes, fs, band, level, window_seconds, noise=None):
    """Generate noise band with time-variant level.

    :param datetimes: Datetimes for which a sample is calculated.
    :param fs: Sample frequency.
    :param level: Sound pressure level.
    :param band: Octaveband.
    """
    nsamples = len(datetimes)
    if (band.upper < fs/2.0).any():
        level = _interpolate_and_fill(level, datetimes, fs, window_seconds)
        noise = noise if noise is not None else Signal(pink(nsamples), fs)
        return np.squeeze(noise.bandpass_frequencies(band, purge=False)[1]).calibrate_to(level)
    else:
        return Signal(np.zeros(nsamples), fs)


def _generate_tone_zipped(parameters):
    return _generate_tone(*parameters)

def _generate_noise_zipped(parameters):
    return _generate_noise(*parameters)

def _generate_emission_signal(datetimes, tones_levels, tones_frequencies, tones_bandwidth, noise_levels, fs=44100., noise_fraction=3, window=3.0, tones_seed=100, noise_seed=100, mapfunc=map):
    """Generate emission signal.

    Generate an emission signal from given features. Values are linearly interpolated at the times specified by `datetimes`.
    Next, a smoothing filter is applied.

    :param datetimes: Datetimes for which a sample is calculated.
    :param fs: Sample frequency.
    :param tone_levels: DataFrame with tone levels
    :param tone_frequencies: DataFrame with tone frequencies
    :param noise_levels: DataFrame with noise levels. Rows are samples, columns are different frequencies.
    :param noise_bands: Iterable with noise bands belonging to centerfrequencies.
    :param mapfunc: Map function. By default the built-in map is used. Map function should allow multiple iterables.
    :returns: Emission signal
    :rtype: :class:`acoustics.Signal`
    """
    logger.info("generate_emission_signal: Generating emission signal.")
    nsamples = len(datetimes)
    synthesis = Signal(np.zeros(nsamples), fs)

    # Synthesize tones
    logger.info("generate_emission_signal: Generating {} tonal components.".format(str(len(tones_levels.columns))))
    tones_state = np.random.RandomState(tones_seed)
    ntones = len(tones_frequencies.columns)
    phases = tones_state.randn(ntones)
    freqs = (x[1] for x in tones_frequencies.items())
    levels = (x[1] for x in tones_levels.items())
    bandwidths = (x[1] for x in tones_bandwidth.items())
    parameters = ((datetimes, fs, frequency, level, bandwidth, window, phase) for frequency, level, bandwidth, phase in zip(freqs, levels, bandwidths, phases))
    synthesis += sum(mapfunc(_generate_tone_zipped, parameters))

    # Synthesize noise
    logger.info("generate_emission_signal: Generating {} noise components.".format(str(len(noise_levels.columns))))
    noise_state = np.random.RandomState(noise_seed)
    noise = Signal(pink(nsamples, state=noise_state), fs)
    nominals, levels = zip(*noise_levels.items())
    bands = (OctaveBand(center=nominal, fraction=noise_fraction) for nominal in nominals)
    parameters = ((datetimes, fs, band, level, window, noise) for band, level in zip(bands, levels))
    synthesis += sum(mapfunc(_generate_noise_zipped, parameters))

    logger.info("generate_emission_signal: Emission signal generated.")
    return synthesis


def get_synthesis(noise, tones, datetimes, settings_features, settings):
    """Get emission synthesis.

    :param noise: Noise features.
    :param tones: Tone features.
    :param datetimes: Datetimes of ...
    :param settings_features: Settings used for computing the features.
    :param settings: Settings to be used for the synthesis.
    :returns: Tuple of emission synthesis and settings that were used.

    """

    # Settings
    settings = recursive_mapping_update(SETTINGS.copy(), settings)

    # Pivot features so we get the features as function of time
    noise_levels, tones_frequencies, tones_levels, tones_bandwidth = _features_as_function_of_time(noise, tones)

    # Synthesise emission
    with Pool() as pool:
        synthesis = _generate_emission_signal(datetimes, tones_levels, tones_frequencies, tones_bandwidth, noise_levels,
                                              fs=settings['fs'],
                                              noise_fraction=settings_features['features']['noise_fraction'],
                                              window=settings['window'],
                                              mapfunc=pool.map)
    synthesis = pd.Series(synthesis, index=datetimes)

    return synthesis, settings

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file_in_noise", type=str)
    parser.add_argument("file_in_tones", type=str)
    parser.add_argument("file_in_settings_features", type=str)
    parser.add_argument("file_in_settings", type=str)
    parser.add_argument("file_in_position", type=str)
    parser.add_argument("file_out_settings", type=str)
    parser.add_argument("file_out_audio", type=str)
    parser.add_argument("file_out_position", type=str)
    args = parser.parse_args()

    # Load settings
    settings = jsonload(args.file_in_settings)

    # Load features
    noise, _ = h5load(args.file_in_noise)
    tones, _ = h5load(args.file_in_tones)
    settings_features = jsonload(args.file_in_settings_features)

    # Datetimes for synthesis
    position, _ = h5load(args.file_in_position)
    datetimes_emission = position.index
    #datetimes = noise_levels.index # Doesn't matter whether we take from tones or noise

    synthesis, settings = get_synthesis(noise, tones, datetimes_emission, settings_features, settings)

    h5save(args.file_out_audio, synthesis, {'fs':settings['fs']})
    h5save(args.file_out_position, position)  # This should be exactly the same as emission (backpropagated) position!!
    jsonsave(args.file_out_settings, settings)


if __name__ == '__main__':
    main()
