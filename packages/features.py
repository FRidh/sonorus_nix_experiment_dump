# Module with functions for extracting features from backpropagated recording
import argparse
import numpy as np
import pandas as pd
import json
import toolz
import itertools
from acoustics import Signal
from acoustics.standards.iso_1996_2_2007 import Tonality
from acoustics.signal import OctaveBand
from acoustics.decibel import dbsum
from h5store import *
from sonorus import receiver_position as get_receiver_position
from sonorus import get_receiver, recursive_mapping_update

from concurrent.futures import ProcessPoolExecutor as Pool

from sonorus import create_logger

logger = create_logger(__name__)

SETTINGS = {
    'window' : 1.0, # Analysis window in seconds.
    'lowpass': 10000, # Lowpass reverted signal.

    'fundamental' : { # Parameters relevant for estimation of the fundamental
        'naverages' : 10, # Amount of averages for assessment of tones.
        'tsc'       : 1.0, # Tone seeking criterion
        'possible'  : list(np.arange(64., 80., 0.1)), # Possible values of the fundamental frequency
        'max_combinations': 200,
        'max_freq' : 0.5, # Maximum frequency of tones to use for estimating fundamental.
        },
    'features' : {
        'nblades': None, # Amount of blades fan.
        'correction_tone' : -6.0, # Correction of tonal components. By default -3.0 for ground reflection.
        'correction_noise': -3.0, # Correction of noise components. By default -3.0 for ground reflection.
        'bandwidth_fraction': 0.2, # Search for tone in fundamental*bandwidth_fraction.
        'max_freq' : 0.5, # Maximum frequency/sample_frequency to look for features.
        'df': 5.0, # Minimum frequency resolution
        'noise_fraction' : 3, # Fraction of fractional-octaves noise.
        },
    }


# Geometry stuff

def source_velocity(source_position, receiver_position, fs):
    """Velocity of source."""
    return self.source_position.apply(np.gradient).mul(fs)

def source_speed(source_position, receiver_position, fs):
    """Speed of source."""
    velocity = source_velocity(source_position, receiver_position, fs)
    return pd.Series(np.linalg.norm(velocity, axis=1), index=velocity.index)

def distance_receiver_to_source(source_position, receiver_position):
    """Distance vector from source to receiver."""
    return source_position - receiver_position

def distance_source_to_receiver(source_position, receiver_position):
    """Distance vector from receiver to source."""
    return -distance_source_to_receiver(source_position, receiver_position)

def distance_scalar(source_position, receiver_position):
    """Scalar distance between source and receiver."""
    distance_components = distance_receiver_to_source(source_position, receiver_position)
    return pd.Series(np.linalg.norm(distance_components, axis=1), index=distance_components.index)

def orientation_receiver_to_source(source_position, receiver_position):
    """Orientation vector from receiver to source."""
    distance = distance_scalar(source_position, receiver_position)
    return distance_receiver_to_source(source_position, receiver_position).div(distance, axis=0)

def orientation_source_to_receiver(source_position, receiver_position):
    """Orientation vector from source to receiver."""
    return -orientation_receiver_to_source(source_position, receiver_position)

def initial_distance(source_position, receiver_position):
    return distance(source_position, receiver_position)


def determine_n1(signal, frequency_range, remove_trend=True):
    """"Determine N1 frequency

    :param signal: Signal
    :param fs: Sample frequency
    :param frequency_range: Frequency range in which to determine N1

    """
    # Determine cepstrum, take 20*log10(C), store as pandas series
    cep = pd.Series(10.0*np.log10(complex_cepstrum(signal, signal.fs)[0]**2.0), index=signal.times())
    # Select the quefrency range
    selection = cep.ix[1./frequency_range[1]:1./frequency_range[0]]
    # Take out the small but noticeable trend. Approximate it as linear.
    if remove_trend:
        selection = pd.Series(detrend(selection), index=selection.index)
    # N1 is the highest peak.
    N1 = 1.0 / selection.argmax()
    return N1


def harmonics(fundamental, fs):
    """Return harmonics of given N1.
    :param fundamental: Fundamental frequency.
    :param fs: Sample frequency.
    :returns: Array with harmonics.
    """
    return np.arange(fundamental, fs/2.0, fundamental)

#def get_n1_harmonics(signal, frequency_range_fundamental, remove_trend=True):
    #"""Return array with harmonics.
    #"""
    #N1 = determine_n1(signal, frequency_range_fundamental, remove_trend=remove_trend)
    #return harmonics(N1, signal.fs)


def reshape_and_truncate(signal, fs=None, blocks=None, samples=None, duration=None):
    """Reshape signal into specified amount of blocks, or in blocks of a certain duration.
    """
    if duration:
        samples = int(duration * fs)
    if samples:
        blocks = signal.shape[-1]//samples
    return signal[..., :( (signal.shape[-1] // blocks) * blocks )].reshape((blocks, -1))

# ---------------------------- Determine features -----------------------------------

def _minimize(harmonics, possible_fundamentals):
    """Determine optimal fundamental."""
    mods = (harmonics[None,...] / possible_fundamentals[...,None]) %1
    rounded_mods = np.round(mods)
    rounded_mods_norm = rounded_mods / rounded_mods.max(axis=-1)[...,None]
    errors = np.sum( (np.abs(rounded_mods - mods)/0.5 * (rounded_mods_norm.max(axis=-1)[...,None]-rounded_mods_norm))**(0.5) , axis=1)

    errors[np.isnan(errors)] = np.inf

    #errors = ne.evaluate("sum(abs(rounded_mods - mods)**2.0, axis=1)")
    return possible_fundamentals[errors.argmin()], errors.min()

def _find_fundamental(harmonics, possible_fundamentals, max_combinations):
    """Find estimate of fundamental frequency given a set of measured harmonics.

    :param harmonics: Harmonics. Unknown which order. Might be inaccurate or not even a harmonic at all.
    :param possible_fundamentals: Allowed values for the fundamental
    :returns: (fundamental, error, harmonics_used)

    """
    logger.debug("_find_fundamental: Searching for fundamental given {} possible harmonics {}.".format(len(harmonics), harmonics))
    initial_fundamental_estimate, initial_error = _minimize(harmonics, possible_fundamentals)
    initial_harmonics_estimate = np.round(harmonics/initial_fundamental_estimate).astype(int)

    # If none of the tones think they're the same harmonic
    if toolz.isdistinct(initial_harmonics_estimate):
        # Then we're done here
        logger.debug("_find_fundamental: No duplicate harmonics.")
        return initial_fundamental_estimate, initial_error, harmonics
    # Else we have to fix this
    else:
        # Let's group the tones by harmonic number
        logger.debug("_find_fundamental: Several duplicate harmonics, checking combinations.")
        grouped = toolz.groupby(lambda x: np.round(x/initial_fundamental_estimate).astype('int'), harmonics)
        grouped_values = [list(value) for value in grouped.values()]
        #logger.info("Grouped values: " + str(grouped_values))
        # All results will be put in a list. Let's begin by adding our initial estimates.
        results = [(initial_fundamental_estimate, initial_error, harmonics)]

        # For each combination of (duplicate) harmonics
        for i, combination in enumerate(itertools.product(*grouped_values)):
            if i < max_combinations:
                # Estimate the fundamental and the error
                #logger.info("Combination: " + str(combination))
                estimate, error = _minimize(np.array(combination), possible_fundamentals)
                results.append((estimate, error, combination))
            else:
                break
        logger.debug("_find_fundamental: All combinations checked. Determining best result.")

        # Now we have estimates and errors for all combinations including our initial estimate.
        results = pd.DataFrame(results, columns=['fundamental', 'error', 'harmonics'])
        # The best result is the result with the smallest error
        best_result = results.ix[results.error.argmin()]
        return best_result.fundamental, best_result.error, np.array(sorted(best_result.harmonics))


def determine_fundamental(signal, max_freq=0.5, possible=None, naverages=10, tsc=1.0, max_combinations=200):
    """Determine fundamental frequency.

    Determine the fundamental frequency using ISO 1996-2:2007 tonality.

    :param signal: Signal
    :param max_freq: Search for tones up to this frequency.
    :param fundamentals: Possible values of the fundamental frequency.
    :param tonality_bins: Amount of bins to use for the power spectrum.
    :param tsc: Tone seeking criterion.
    :returns: Estimate of the fundamental frequency.
    """
    possible_fundamentals = np.array(possible)
    if possible_fundamentals is None:
        raise ValueError("possible_fundamentals needs to be an array of values to choose from")
    tonality_bins = int(np.floor(signal.fs / naverages))

    logger.info("determine_fundamental: Determining fundamental frequency using tones below {} Hz.".format(max_freq*signal.fs))
    tonality = Tonality(signal.correlate()[:signal.samples], signal.fs, nbins=tonality_bins, tsc=tsc)
    tonality.determine_noise_pauses().analyse()
    logger.info("determine_fundamental: Noise pauses and initial tones determined.")
    harmonics = np.array(tonality.results_as_dataframe()['center'])
    harmonics = harmonics[harmonics < max_freq*signal.fs]
    #print("Amount of harmonics: {}".format(len(harmonics)))
    estimated_fundamental, error, harmonics = _find_fundamental(harmonics, possible_fundamentals, max_combinations)
    return estimated_fundamental


#def _noise_regression_tones(spectrum, tones):
    #"""Perform linear regression on noise lines. Replace tone line values by
    #values calculated from regression.
    #"""
    #for tone in tones:
        ###start, stop = tone.noise_pause.start, tone.noise_pause.end
        ###slicer = slice(tone.noise_pause.start, tone.noise_pause.end)
        ##slicer = tone._tone_lines
        ##slope = tone.critical_band.regression_slope
        ##intercept = tone.critical_band.regression_intercept
        ##newlines = spectrum.ix[slicer].index * slope + intercept
        ##yield tone_level_difference = dbsub(tone.tone_level, float(dbsum(newlines)))
        #if int(np.round(tone.center//74.))== 36:
            #pass
            ##print(tone.bandwidth_3db, tone.center)
            ##print("Tone", tone.tone_level, float(dbsum(newlines)), dbsub(tone.tone_level, float(dbsum(newlines))))
        ##tone.tone_level = float(dbsum(newlines))

def _override_bpf_harmonics(tones, fundamental, spectrum, line_classifier, bandwidth, nblades):
    """Override values for blade passing frequency harmonics.

    :param tones: Iterable of tones.
    :param fundamental: Fundamental frequency.
    :param spectrum: Spectrum.
    :param line_classifier: Line classifier of Tonality.
    :param bandwidth: Bandwidth assigned for tone.
    """
    for tone in tones:
        # If blade passing frequency harmonics...
        if not int(np.round(tone.center/fundamental)) % nblades:
            logger.debug("override_bpf_harmonics: Found bpf at {}".format(tone.center))
            # Set new frequency lines in line designator
            lines = line_classifier[tone.center-bandwidth/2.0:tone.center+bandwidth/2.0]
            line_classifier[lines.index] = 'tone'
            # Calculate new level
            tone.tone_level = dbsum(spectrum[lines.index])
            # Set bandwidth of tone
            tone.bandwidth_3db = bandwidth

def _determine_levels(signal, fundamental, settings):
    """Determine level of tones and level of 1/3-octave bands.

    Determine the level of tones and fractional-octaves using ISO 1996-2:2007 tonality.

    :param signal: Signal
    :param fundamental: Estimate of fundamental frequency.
    :param settings: Analysis Features settings.
    :returns: Two DataFrames, first describing tones, second describing noise.

    This function is supposed to be used over a small interval, for example 1 second.
    """
    logger.debug("_determine_levels: Determine features below {}".format(settings['max_freq']*signal.fs))

    bandwidth_fraction = settings['bandwidth_fraction']
    nblades = settings['nblades']

    tonality = Tonality(signal, signal.fs, nbins=int(np.ceil(signal.fs/settings['df'])))
    tonality.force_tone_without_pause = True
    #tonality.force_bandwidth_criterion = True
    #tonality.regression_range_factor = 4.0

    estimated_harmonics = np.arange(fundamental, settings['max_freq']*signal.fs, fundamental)
    #bpf_harmonics_indices = ~((estimated_harmonics/fundamental)%nblades).astype(bool)
    #buzz_saw_harmonics = estimated_harmonics[~bpf_harmonics_indices]
    #bpf_harmonics = estimated_harmonics[bpf_harmonics_indices]

    # possible noise pause start and stop indices. Note the factor 1/8!!!
    start = (np.round((estimated_harmonics - fundamental*bandwidth_fraction) / tonality.frequency_resolution)).astype(int)
    end = (np.round((estimated_harmonics + fundamental*bandwidth_fraction) / tonality.frequency_resolution) -1).astype(int)

    # Prevents out of bounds
    valid_indices = end <  len(tonality.spectrum) - 2
    start = start[valid_indices]
    end = end[valid_indices]

    #bpf_start = start[nblades-1::nblades]
    #bpf_end = end[nblades-1::nblades]

    #start = (np.round((buzz_saw_harmonics - fundamental/8.0) /tonality.frequency_resolution)).astype(int)
    #end = (np.round((buzz_saw_harmonics + fundamental/8.0) /tonality.frequency_resolution) -1).astype(int)

    #bpf_start = (np.round((bpf_harmonics - fundamental/bandwidth_fraction) /tonality.frequency_resolution)).astype(int)
    #bpf_end = (np.round((bpf_harmonics + fundamental/bandwidth_fraction) /tonality.frequency_resolution) -1).astype(int)

    #start = np.concatenate((start, (np.round((bpf_harmonics - fundamental/3.0) /tonality.frequency_resolution)).astype(int) ))
    #end = np.concatenate((end, (np.round((bpf_harmonics + fundamental/3.0) /tonality.frequency_resolution) -1).astype(int) ))

    # Set noise pauses to estimated values of the harmonics
    tonality._set_noise_pauses(zip(start, end)).analyse()

    # Set tone lines to values obtained from noise regression
    #_noise_regression_tones(tonality.spectrum, tonality.tones)

    # Handle blade passing frequency harmonics differently.
    if nblades:
        _override_bpf_harmonics(tonality.tones, fundamental, tonality.spectrum, tonality.line_classifier, fundamental*bandwidth_fraction, nblades)

    tonality.line_classifier[tonality.line_classifier!='tone'] = 'noise'

    results = tonality.results_as_dataframe()

    #print(len(estimated_harmonics[valid_indices]), len(results.tone_level), len(results.bandwidth_3db), len(results.center))
    print(len(results))

    # Tone level as function of frequency
    tones = pd.DataFrame({
                          #'center' : estimated_harmonics[valid_indices],
                          'center' : results.center,
                          'level' : results.tone_level - 1.8, # Hanning window correction
                          'bandwidth' : results.bandwidth_3db,
                          'harmonic' : np.round(results.center / fundamental).astype(int),
                          }).set_index('center')

    # Level and designator as function of frequency
    spectrum = pd.DataFrame({'level': tonality.spectrum, 'mark': tonality.line_classifier})
    noise_spectrum = spectrum[spectrum.mark=='noise'] # We disregard noise within the noise pause!
    del spectrum

    # Frequency bins for groupby
    noise_bands = OctaveBand(fstart=10.0, fstop=signal.fs/2.0, fraction=settings['noise_fraction'])
    fbins = np.append(noise_bands.lower, noise_bands.upper[-1])

    noise = noise_spectrum.level.groupby(pd.cut(noise_spectrum.index, fbins)).apply(dbsum)

    noise = pd.DataFrame({'level': noise - 1.8, # Hanning window correction
                          'center': noise_bands.center,
                          'lower': noise_bands.lower,
                          'upper': noise_bands.upper,
                          'fraction' : noise_bands.fraction,
                          'nominal' : noise_bands.nominal,
                          }).set_index('center')

    # Correct for ground reflection.
    tones['level'] += settings['correction_tone']
    noise['level'] += settings['correction_noise']

    return tones, noise


def _extract_tones_and_noise(signal, settings):
    """Determine fundamental frequency, tone features and noise features.

    :param signal: Signal
    :param settings: Analysis settings.
    :returns: Fundamental frequency, and two DataFrames, one describing tones, and a second describing noise.

    This function is supposed to be used over a small interval, for example 1 second.
    """
    # Determine fundamental
    #if len(signal) < tonality_bins:
        #fundamental = +np.inf
    #else:
    fundamental = determine_fundamental(signal, **settings['fundamental'])

    logger.debug("_extract_tones_and_noise: Fundamental determined: {}".format(fundamental))

    # Extract features
    #ob = OctaveBand(fstart=10.0, fstop=signal.fs/2.0, fraction=fraction)
    tones, noise = _determine_levels(signal, fundamental, settings['features'])

    logger.debug("_extract_tones_and_noise: Tone and noise levels determined.")

    return fundamental, tones, noise


def _features_for_group(group, fs, orientation, distance, settings):
    """Determine features over a short time-interval using :func:`_extract_tones_and_noise` and add several more columns to the obtained features.

    :param group: Selected interval over which to determine a single set of features. A group typically represents one second.
    :param fs: Sample frequency.
    :param orientation: Orientation.
    :param distance: Distance.
    :param settings: Analysis settings.
    :returns: Two dataframes, one describing noise, and another describing tones. Rows are samples (time), columns represent tones/harmonics or noise bands.
    """
    logger.debug("_features_for_group: Extracting features of group.")
    datetime = group.index[len(group)//2]
    #datetime = group.index[0]

    # If our signal is only zeros, stop.
    #if (group < np.finfo(float).eps).all():
        #return None
    fundamental, tones, noise = _extract_tones_and_noise(Signal(group, fs), settings)

    logger.debug("_features_for_group: Constructing tables.")
    # Build tones table
    tones = tones.reset_index().rename(columns={'center':'frequency'})
    tones = tones.assign(
                         datetime=datetime,
                         distance=distance.asof(datetime),
                         x=orientation.x.asof(datetime),
                         y=orientation.y.asof(datetime),
                         z=orientation.z.asof(datetime),
                         )

    # Build noise table
    noise = noise.reset_index()
    noise = noise.assign(
                         datetime=datetime,
                         distance=distance.asof(datetime),
                         x=orientation.x.asof(datetime),
                         y=orientation.y.asof(datetime),
                         z=orientation.z.asof(datetime),
                         )
    return noise, tones

def _features_for_group_zipped(parameters):
    return _features_for_group(*parameters)


def extract_features(source_position, receiver_position, signal, datetimes, fs, settings, mapfunc=map):
    """Extract features of a given event as function of time, with a time-resolution given by `interval`.

    :param source_position: Source position. DataFrame
    :param receiver_position: Receiver position. Array.
    :param signal: Backpropagated sound. Series
    :param datetimes: Series
    :param fs: Sample frequency.
    :param settings: Analysis settings.
    :param mapfunc: Map function. By default the built-in map is used.
    :returns: Two Dataframes, one describing noise, another tones/harmonics.

    """
    logger.info("extract_features: Extracting features...")
    ### Obtain actual event and backpropagate.
    ##if not hasattr(eventanalysis, 'reverted_as_signal'):
        ##eventanalysis.revert()

    # Low-pass because of noise amplification in backpropagation
    logger.debug("extract_features: Low-pass signal.")
    signal = Signal(signal, fs).lowpass(settings['lowpass'])
    logger.debug("extract_features: Create pandas series.")
    signal = pd.Series(signal, index=datetimes[:signal.samples]) # Only part where we have signal (Doppler)

    logger.debug("extract_features: Dividing signal in chunks.")
    groups = (items[1] for items in signal.groupby(pd.TimeGrouper("{0:.0f}s".format(settings['window'])))) # pandas format is '1s'.

    logger.debug("extract_features: Finished grouping. Extracting now...")

    # Note that at this point we're not storing this.
    orientation = orientation_source_to_receiver(source_position, receiver_position)
    distance = distance_scalar(source_position, receiver_position)

    #_features_for_group_partial = partial(_features_for_group, event, possible_fundamentals, noise_fraction, maxfreq)
    parameters = ((group, fs, orientation, distance, settings) for group in groups)
    results = mapfunc(_features_for_group_zipped, parameters)
    #results = (result for result in results if result is not None)
    noise, tones = zip(*list(results))
    noise = pd.concat(noise)
    tones = pd.concat(tones)
    logger.info("extract_features: Finished extracting features.")
    return noise, tones


def get_features(receiver_position, source_position, reverted, fs, settings):

    # Settings
    settings = recursive_mapping_update(SETTINGS.copy(), settings)

    # Extract features
    with Pool() as pool:
        noise, tones = extract_features(source_position, receiver_position, reverted, source_position.index, fs, settings=settings, mapfunc=pool.map)

    return noise, tones, settings


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("receiver", type=str)
    parser.add_argument("file_in_backpropagated", type=str)
    parser.add_argument("file_in_source_position", type=str)
    parser.add_argument("file_in_settings", type=str)
    parser.add_argument("file_out_noise", type=str)
    parser.add_argument("file_out_tones", type=str)
    parser.add_argument("file_out_settings", type=str)
    args = parser.parse_args()

    # Load other files
    settings = jsonload(args.file_in_settings)
    source_position, meta = h5load(args.file_in_source_position)
    receiver_position = get_receiver_position(get_receiver(args.receiver))
    signal, meta = h5load(args.file_in_backpropagated)
    fs = meta['fs']

    # Compute
    noise, tones, settings = get_features(receiver_position, source_position, signal, fs, settings)

    # Save features
    h5save(args.file_out_noise, noise, {})
    h5save(args.file_out_tones, tones, {})
    jsonsave(args.file_out_settings, settings)


if __name__ == '__main__':
    main()
