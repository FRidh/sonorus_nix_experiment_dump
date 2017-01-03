# Get audio and position given an event, receiver, and start and stop times.
# The data is stored in a pd.DataFrame, and the sample frequency is the attribute fs.
# We pickle the data.

import pandas as pd
from blaze import Data
from odo import into, odo
import argparse
import h5py
import numpy as np
import os
from h5store import h5save

from sonorus import DB_BLAZE, FILE_RECORDINGS, DB_TABLE_RECORDINGS

#DB_BLAZE = 'sqlite:///' + "@database@"
#FILE_RECORDINGS = "@audio@"
#DB_TABLE_RECORDINGS = 'recording'


def relative_seconds_to_datetime(event, time, position=None):
    """Convert from seconds relative to `datetime_nearest` to `datetime.

    :param event: Event name
    :param time: Relative start time in seconds.
    :param position: Position. Can be either `source_time` or any of the receivers.

    .. seealso:: :func:`relative_seconds_to_datetimes`

    """
    if type(time) is float:
        if position is None:
            datetime_nearest = get_event(event).datetime_nearest_receiver
        else:
            datetime_nearest = get_datetime_nearest_at_position(event, position)
        try:
            time = datetime_nearest + pd.Timedelta(seconds=time)
        except TypeError:
            raise TypeError("Invalid start time type.")
    return time


def relative_seconds_to_datetimes(event, start, stop, position=None):
    """Convert from seconds relative to `datetime_nearest` to `datetime`.

    :param event: Event name
    :param start: Relative start time in seconds.
    :param stop: Relative stop time in seconds.
    :param position: Position. Can be either `source_time` or any of the receivers.

    """
    start = relative_seconds_to_datetime(event, start, position=position)
    stop = relative_seconds_to_datetime(event, stop, position=position)
    return start, stop


def get_datetime_nearest_at_position(event, receiver):
    """Get datetime_nearest of event at receiver.
    """
    data = Data(DB_BLAZE)
    datetime = odo(data.datetime_nearest[data.datetime_nearest.event==event][['event', receiver]], pd.DataFrame)[receiver].loc[0]
    return datetime


def get_recordings(receiver, datetime_start, datetime_stop):
    """Get recordings.

    :param receiver: Name of receiver.
    :param datetime_start: Datetime start.
    :param datetime_stop: Datetime stop.
    :returns: Returns a list of tuples of format ('recording_name', 'datetime_start', 'datetime_stop') ordered by 'datetime_start'.
    """
    datetime_start =  pd.to_datetime(datetime_start)
    datetime_stop = pd.to_datetime(datetime_stop)

    data = Data(DB_BLAZE + '::' + DB_TABLE_RECORDINGS)
    selection = data[ (data.datetime_stop >= datetime_start) & (datetime_stop >= data.datetime_start) & (data.receiver==receiver) ][['name', 'datetime_start', 'datetime_stop']].sort('datetime_start')

    return into(list, selection)


def get_audio(receiver, datetime_start, datetime_stop):
    """Get audio.

    :param datetime_start: Datetime audio begins.
    :param datestime_stop: Datetime audio stops.
    :param receiver: Name of receiver.
    :returns: Tuple consisting of sample frequency and an array with the signal.

    """
    items = get_recordings(receiver, datetime_start, datetime_stop)

    if not items:
        raise ValueError("No audio available for specified datetimes and receiver.")

    datetime_start =  pd.to_datetime(datetime_start)
    datetime_stop = pd.to_datetime(datetime_stop)

    data = h5py.File(FILE_RECORDINGS, 'r')
    #data = Data(data) # Use blaze

    sample_frequencies = list()
    lengths = list()
    out = np.array([])
    for name, start, stop in items:
        s = data[os.path.join('recordings', name)]
        fs = s.attrs['fs_gps'] # Sample frequency based on GPS time.
        samples = s.len()

        if datetime_start <= start:
            range_start = 0
        else:
            range_start = int(round( (datetime_start - start).total_seconds() * fs ))

        if datetime_stop >= stop:
            range_stop = samples
        else:
            range_stop = int(round( (datetime_stop - start).total_seconds() * fs ))

        with s.astype('float64'):
            out = np.concatenate((out, s[range_start:range_stop]))
        sample_frequencies.append(fs)
        lengths.append(range_stop-range_start)

    try:
        fs = np.average(sample_frequencies, weights=lengths)
    except ZeroDivisionError:
        fs = 0.0

    return fs, out


def get_event_position(event, datetime_start, datetime_stop, outside=True):
    """Get position of aircraft given an event.

    :param event: Event name
    :param datetime_start: Start datetime
    :param datetime_stop: Stop datetime
    :param outside: Select single sample before and after selection.

    When `outside=True` a single sample before and after the requested selection is taken.
    This allows interpolation. Note that `datetime_start` and `datetime_stop` are
    not affected by this.

    """
    data = Data(DB_BLAZE).position

    track = into(pd.DataFrame, data[data.event == event].sort()[['datetime', 'x', 'y', 'z']])
    track.set_index('datetime', inplace=True)

    track_datetime_start = track.index.min()
    track_datetime_stop = track.index.max()

    if track_datetime_start > datetime_start:
        raise ValueError("Requested start occurs before tracking initiated.")
    if track_datetime_stop < datetime_stop:
        raise ValueError("Requested stop occurs after tracking stopped.")

    # Select correct part of track. Could have been done with blaze as well...
    # This does not include two samples outside of selection. This causes problems with interpolation, requiring extrapolation
    # Instead, select single sample outside these boundaries as well.
    #track = track[ (track.index >= datetime_start) & (track.index <= datetime_stop) ]

    selection = (track.index >= datetime_start) & (track.index <= datetime_stop)
    start_index = selection.argmax()
    stop_index = len(selection) - selection[::-1].argmax()
    if outside:
        start_index -= 1
        stop_index += 1
    track = track[start_index:stop_index]
    return track#, datetime_start, datetime_stop


def _datetimes_range(datetime_start, datetime_stop, fs):
    """Generate a datetime index given from `datetime_start` to `datetime_stop` with a sample every 1/`fs`."""
    td = datetime_stop - datetime_start
    samples = round(  td.total_seconds() * fs )
    times = pd.to_datetime(np.datetime64(datetime_start) + np.timedelta64(int(round((1./fs)*(1e+9))), 'ns') * np.arange(samples) )
    return times

def get_audio_and_position(event, receiver, datetime_start, datetime_stop):
    """Get audio and position given an event and receiver.

    :param event: Event name
    :param receiver: Receiver name
    :param datetime_start: Start datetime
    :param datetime_stop: Stop datetime
    :returns: Returns a tuple (fs, data) where data is a DataFrame with columns ['signal', 'x', 'y', 'z'].

    Get recording and position of the aircraft. Resample position to sample frequency of the recording.
    """
    track = get_event_position(event, datetime_start, datetime_stop, outside=True)

    fs, signal = get_audio(receiver, datetime_start, datetime_stop)
    #times = pd.to_datetime(np.datetime64(datetime_start) +
                           #np.timedelta64(int(round((1./fs)*(1e+9))), 'ns') * np.arange(len(signal)) )
    times = _datetimes_range(datetime_start, datetime_stop, fs)

    signal = pd.DataFrame({'signal':signal, 'datetime':times})
    signal.set_index('datetime', inplace=True)

    #out = pd.concat([signal, positions]).sort_index()
    out = signal.join(track, how='outer')

    out.x.interpolate('linear', inplace=True)
    out.x.fillna(method='backfill', inplace=True)
    out.y.interpolate('linear', inplace=True)
    out.y.fillna(method='backfill', inplace=True)
    out.z.interpolate('linear', inplace=True)
    out.z.fillna(method='backfill', inplace=True)

    out = out.loc[times]
    return fs, out


def get_recording(event, receiver, start, stop):
    """Get audio and position of an event at a receiver.

    :param event: Event
    :param receiver: Receiver
    :param start: Relative start time in seconds
    :param stop: Relative stop time in seconds
    :returns: Tuple consisting of recording, position and sample frequency

    """
    datetime_start, datetime_stop = relative_seconds_to_datetimes(event, start, stop, position=receiver)
    fs, audio_and_position = get_audio_and_position(event, receiver, datetime_start, datetime_stop)
    recording = audio_and_position.signal
    position = audio_and_position[['x', 'y', 'z']]
    return recording, position, fs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("event", type=str)
    parser.add_argument("receiver", type=str)
    parser.add_argument("start", type=float)
    parser.add_argument("stop", type=float)
    parser.add_argument("recording", type=str)
    parser.add_argument("position", type=str)
    args = parser.parse_args()

    recording, position, fs = get_recording(args.event, args.receiver, args.start, args.stop)

    h5save(args.recording, recording, {'fs':fs})
    h5save(args.position , position, {'fs':fs})


if __name__ == '__main__':
    main()
