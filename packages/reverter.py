import numpy as np
import pandas as pd
import json
import argparse
from h5store import *
from sonorus import get_receiver, get_event_atmosphere
from auraliser import Reverter
from acoustics import Signal
from sonorus import create_logger, recursive_mapping_update

logger = create_logger(__name__)

SETTINGS = {
}

def _get_reverted_audio(event, receiver, recording, track, fs, settings=None):
    """Get reverted audio given an event and receiver.

    :param filename: Filename of file with audio and position.
    :param settings: Settings to pass on to :class:`Auraliser` object.
    :returns: reverted signal and recorded signal

    """
    receiver = get_receiver(receiver)
    emission_signal = Signal(recording, fs)
    receiver_position = (receiver.x, receiver.y, receiver.height_absolute)
    atmosphere = get_event_atmosphere(event)
    reverted_signal = _revert(emission_signal, track, receiver_position, atmosphere, settings)

    #print(len(emission_signal), len(reverted_signal), len(track))
    return reverted_signal

# merge these two functions?

def _revert(signal, source_position, receiver_position, atmosphere, settings):
    """Revert emission signal.

    :param: Signal is an instance of :class:`acoustics.Signal`.
    :param source_position: Source position. Dataframe with columns ['x', 'y', 'z'] and `len(source_position)==len(signal)`.
    :param receiver_position: Tuple (x, y, z)
    :param atmosphere: Instance of :class:`acoustics.atmosphere.Atmosphere`.
    :param settings: Auraliser/Reverter settings.
    """
    logger.info("_revert: Reverting signal.")
    reverter_source_position = np.array(source_position)
    reverter_receiver_position = np.tile(np.asarray(receiver_position), (len(signal), 1))
    reverter = Reverter(source=reverter_source_position,
                        receiver=reverter_receiver_position,
                        atmosphere=atmosphere,
                        settings=settings
                        )
    reverted_signal = reverter.revert(signal)
    logger.info("_revert: Finished reverting signal.")
    return reverted_signal


def get_emission(event, receiver, recording, position, fs, settings):
    """Get emission.

    :param event: Event
    :param receiver: Receiver
    :param recording: Audio recorded at receiver
    :param position: Position of source
    :param fs: Sample frequency
    :param settings: Settings

    .. seealso:: :func:`recording.get_recording`
    """

    # Settings
    settings = recursive_mapping_update(SETTINGS.copy(), settings)

    # Backpropagate
    reverted = _get_reverted_audio(event, receiver, recording, position, fs, settings)

    # Because we have a propagation delay the sample are shifted to the left and out of bounds.
    # Therefore, our reverted signal is shorter. The beginning of the signal corresponds
    # to the beginning of our timestamps.
    reverted = pd.Series(reverted, index=recording.index[:reverted.samples])
    position = position.loc[reverted.index]

    return reverted, position, fs, settings

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("event", type=str)
    parser.add_argument("receiver", type=str)
    parser.add_argument("file_in_immission_audio", type=str)
    parser.add_argument("file_in_immission_position", type=str)
    parser.add_argument("file_in_settings", type=str)
    parser.add_argument("file_out_emission_audio", type=str)
    parser.add_argument("file_out_emission_position", type=str)
    parser.add_argument("file_out_settings", type=str)
    args = parser.parse_args()

    # Load audio and position
    recording, meta = h5load(args.file_in_immission_audio)
    track, meta = h5load(args.file_in_immission_position)
    fs = meta['fs']
    settings = jsonload(args.file_in_settings)

    # Compute
    reverted, position, fs, settings = get_emission(args.event, args.receiver, recording, track, fs, settings)

    # Store result
    h5save(args.file_out_emission_audio, reverted, {'fs': fs})
    h5save(args.file_out_emission_position, position, {'fs': fs})
    jsonsave(args.file_out_settings, settings)

if __name__ == '__main__':
    main()
