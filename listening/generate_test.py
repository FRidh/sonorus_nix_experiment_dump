from sonorus import DB_BLAZE
from blaze import Data
from odo import odo

import numpy as np
from acoustics import Signal

from recording import get_recording
from revert import get_emission
from features import get_features
from synthesis import get_synthesis
from auralisation import get_auralisation


def get_recording_and_auralisation(event, receiver, start, stop, settings_backpropagation,
                                   settings_features, settings_synthesis, settings_auralisation):
    """Get recording and auralisation"""

    # Obtain relevant recording
    recording, position_recording, fs = get_recording(event, receiver, start, stop)

    # Backpropagate
    reverted, position_reverted, fs, settings = get_emission(event, receiver, recording, position_recording, fs, settings_backpropagation)

    # Extract features
    noise, tones, settings_features = get_features(receiver_position, position_reverted, reverted, fs, settings_features)

    # Synthesise emission
    synthesis, _ = get_synthesis(noise, tones, datetimes, settings_features, settings_synthesis)

    # Create auralisation
    auralisation, _, _ = get_auralisation(event, receiver, settings_auralisation, position_reverted, synthesis, fs)

    return recording, auralisation


def fade(signal, duration):
    """Fade head of the signal.
    """
    samples = int(duration * signal.fs)
    fade = np.linspace(0., 1., samples)
    signal[0:samples] *= fade
    return signal


def fade_both_sides(stimuli, duration_start, duration_end):
    """Fade both head and tile of the signal.
    """
    # Fade the head first.
    stimuli = fade(stimuli, duration_start)
    # Then mirror the signal, fade the head of this signal, and mirror again
    stimuli = fade(stimuli[::-1], duration_end)[::-1]
    # to obtain a signal with faded head and tail
    return stimuli


def get_events(nevents, aircraft=None, operation=None, receivers=None):
    """
    :param aircraft: List of aircraft names.
    :param operation: List of operation names.
    :param receivers: List of receivers.
    """
    INCLUDE = 1

    d = Data(DB_BLAZE)
    e = d.event

    # Selection of events
    s = e[e.include==INCLUDE and
          e.operation.isin(operation) and
          e.aircraft.isin(aircraft)]

    s = odo(s, pd.Series)

    s.sample(n=nevents)

    pass






