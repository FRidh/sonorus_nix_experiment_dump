import argparse
from auraliser import get_default_settings, mono, Auraliser
from auraliser.generator import Custom
from acoustics.reflection import impedance_attenborough
from geometry import Point
from ism import Model, Wall
from sonorus import get_receiver, get_event_atmosphere, recursive_mapping_update
from h5store import *
from acoustics import Signal
import numpy as np

SETTINGS = {
    'turbulence'    :   {
        'include'   :   True,
        'correlation_length'    :   10.0,
        'mean_mu_squared'   :   3.0e-4,
        },
}


def auralisation_model(receiver, datetimes, position, atmosphere, synthesis, fs, nsamples, settings=None):
    """Create auralisation model.
    """

    #datetimes = datetimes[:nsamples]
    #track = track.ix[datetimes].values

    _settings = get_default_settings()
    if settings:
        recursive_mapping_update(_settings, settings)
    settings = _settings

    receiver = get_receiver(receiver)

    duration = nsamples / fs

    settings['fs'] = fs

    # Create model
    model = Auraliser(duration=duration, settings=_settings, atmosphere=atmosphere)

    # Create source
    src = model.add_source(name='source', position=position.values)
    subsrc = src.add_subsource(name='subsource')
    vsrc = subsrc.add_virtualsource(name='synthesis', signal=Custom(synthesis), level=synthesis.leq())

    # Create receiver
    rcv = model.add_receiver(name='receiver', position=Point(receiver.x, receiver.y, receiver.height_absolute))

    # Create impedance
    flow_resistivity = 2.0e5 # Grass
    frequencies = np.fft.rfftfreq(model.settings['reflections']['ntaps'], 1./fs)
    impedance = np.nan_to_num(impedance_attenborough(frequencies, flow_resistivity))

    # Create ground
    ground_height = receiver.z
    groundcorners1 = [Point(-10000.0+receiver.x, -10000.0+receiver.y, ground_height),
                        Point(+10000.0+receiver.x, -10000.0+receiver.y, ground_height),
                        Point(+10000.0+receiver.x, +10000.0+receiver.y, ground_height),
                        Point(-10000.0+receiver.x, +10000.0+receiver.y, ground_height) ]
    ground1 = Wall(groundcorners1, Point(0.0, 0.0, ground_height), impedance)
    model.geometry.walls = [ground1]

    return model


def get_auralisation(event, receiver, settings, position, synthesis, fs):
    """Get auralisation.

    :param event: Event
    :param receiver: Receiver
    :param settings: Settings to be used for the auralisation.
    :param position: Emission position.
    :param synthesis: Emission synthesis.
    :param fs: Sample frequency.
    :returns: Auralisation, position and settings.
    """

    # Settings
    settings = recursive_mapping_update(SETTINGS.copy(), settings)

    atmosphere = get_event_atmosphere(event)
    datetimes = position.index

    # Perform auralisation
    synthesis = Signal(synthesis, fs)
    nsamples = len(datetimes)
    model = auralisation_model(receiver, datetimes, position, atmosphere, synthesis, fs, nsamples, settings)
    rcv = list(model.receivers)[0]
    auralisation = mono(rcv.auralise()).toarray()
    dsamples = len(datetimes) - len(auralisation)
    auralisation = pd.Series(auralisation, index=datetimes[dsamples:])

    return auralisation, position, settings

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("event", type=str)
    parser.add_argument("receiver", type=str)
    parser.add_argument("file_in_settings", type=str)
    parser.add_argument("file_in_position", type=str)
    parser.add_argument("file_in_synthesis", type=str)
    parser.add_argument("file_out_position", type=str)
    parser.add_argument("file_out_audio", type=str)
    parser.add_argument("file_out_settings", type=str)
    args = parser.parse_args()

    # Load synthesis, position, datetimes and atmosphere
    settings = jsonload(args.file_in_settings)
    position, _ = h5load(args.file_in_position)
    synthesis, meta = h5load(args.file_in_synthesis)
    fs = meta['fs']

    auralisation, position, settings = get_auralisation(args.event, args.receiver, settings, position, synthesis, fs)

    # Save output
    h5save(args.file_out_audio, auralisation, {'fs':fs})
    h5save(args.file_out_position, position)
    jsonsave(args.file_out_settings, settings)




if __name__ == '__main__':
    main()
