import pandas as pd
import json

"""
Functions to store and load data.
"""

PATH = '/mydata'
"""Default path in hdf5 file."""

def h5save(filename, df, metadata=None, path=PATH):
    """Load a DataFrame with metadata from `filename`."""
    if metadata is None:
        metadata = {}
    with pd.HDFStore(filename, 'w') as store:
        store.put(path, df)
        store.get_storer(path).attrs.metadata = metadata


def h5load(filename, path=PATH):
    """Store a DataFrame with metadata in `filename`."""
    with pd.HDFStore(filename, 'r') as store:
        data = store[path]
        metadata = store.get_storer(path).attrs.metadata
        return data, metadata


def jsonsave(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f)

def jsonload(filename):
    with open(filename, 'r') as f:
        return json.load(f)
