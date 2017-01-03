# Common functions for accessing the database
import numpy as np
from odo import odo
import pandas as pd
import collections
from blaze import Data
from acoustics.atmosphere import Atmosphere
import logging

DB_BLAZE = 'sqlite:///' + "@database@"
FILE_RECORDINGS = "@audio@"

# Names of tables
DB_TABLE_RUNWAYS            = 'runway'
DB_TABLE_RECEIVERS          = 'receiver'
DB_TABLE_CAMERAS            = 'camera'
DB_TABLE_OPERATIONS         = 'operation'
DB_TABLE_RECORDINGS         = 'recording'
DB_TABLE_EVENTS             = 'event'
DB_TABLE_POSITIONS          = 'position'
DB_TABLE_DATETIME_NEAREST   = 'datetime_nearest'

# Fields in tables
DB_TABLE_FIELDS_RUNWAYS     = ['name', 'x', 'y', 'z', 'width', 'length']
DB_TABLE_FIELDS_OPERATIONS  = ['name', 'runway', 'takeoff', 'datetime_nearest_close']
DB_TABLE_FIELDS_CAMERAS     = ['name', 'operation', 'x', 'y', 'z', 'height_relative', 'height_absolute']
DB_TABLE_FIELDS_RECEIVERS   = ['name', 'operation', 'x', 'y', 'z', 'height_relative', 'height_absolute']
DB_TABLE_FIELDS_RECORDINGS  = ['name', 'receiver', 'filename', 'calibration_factor', 'datetime_start', 'datetime_stop']
DB_TABLE_FIELDS_EVENTS      = ['name', 'operation', 'datetime_nearest_receiver', 'aircraft', 'temperature_2m', 'temperature_5cm', 'humidity', 'windspeed', 'pressure', 'include']
DB_TABLE_FIELDS_TRACKING    = ['event', 'datetime', 'x', 'y', 'z']
DB_TABLE_FIELDS_DATETIME_NEAREST = []

# Zero degree Celsius in kelvin.
KELVIN = 273.15

Event = collections.namedtuple('Event', DB_TABLE_FIELDS_EVENTS)
Receiver = collections.namedtuple('Receiver', DB_TABLE_FIELDS_RECEIVERS)

def create_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(handler)
    return logger

def get_event(event):
    """Get event with name.

    :param name: Name of event.
    :returns: Returns a dictionary.

    """
    if isinstance(event, str):
        name = event
    else:
        try:
            name = event.name
        except AttributeError:
            raise ValueError("Cannot get event. Use a string.")
    db = Data(DB_BLAZE)
    selection = odo(db.event[db.event.name==name], pd.DataFrame)
    if selection.empty:
        raise ValueError("Event does not exist.")
    else:
        event = dict(selection.loc[0])
    #event = collections.namedtuple('Event', event.keys())(**event)
    event = Event(**event)
    return event


def get_event_atmosphere(event):
    """Get state of event during event.

    :param event: Name of event.
    :returns: Atmosphere object.

    """
    event = get_event(event)
    atmosphere = Atmosphere(pressure=event.pressure/10.0, # mbar to kPa
                            temperature=KELVIN+event.temperature_2m,
                            relative_humidity=event.humidity/100.0, # between 0.0 and 1.0
                            )
    return atmosphere

def get_receiver(receiver):
    """Get receiver with name. Returns a dictionary."""
    if isinstance(receiver, str):
        name = receiver
    else:
        try:
            name = receiver.name
        except AttributeError:
            raise ValueError("Cannot get receiver. Use a string.")
    db = Data(DB_BLAZE)
    selection = odo(db.receiver[db.receiver.name==name], pd.DataFrame)
    if selection.empty:
        raise ValueError("Receiver does not exist.")
    receiver = dict(selection.loc[0])
    receiver = Receiver(**receiver)
    #receiver = collections.namedtuple('Receiver', receiver.keys())(**receiver)
    return receiver

def receiver_position(receiver):
    return np.array([receiver.x, receiver.y, receiver.height_absolute])


def recursive_mapping_update(d, u):
    """Recursively update a mapping/dict.

    :param d: Target mapping.
    :param u: Source mapping.

    """
    if u is not None:
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                r = recursive_mapping_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
    return d