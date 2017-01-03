{lib, runCommand, python}:

rec {

#   runPythonModule = name: env: runCommand name {buildInputs = [ env ];} ''
#     python -m
#   '';

  # Audio and position of an event as pd.DataFrame
  get_immission = { event, receiver, start, stop }:
    let
      env = python.withPackages(ps: with ps; [ recording ]);
      name = "${event}-${receiver}-immission";
    in runCommand name {
      buildInputs = [ env ];
      outputs = ["out" "audio" "position"];
    } ''
      python -m recording ${event} ${receiver} ${start} ${stop} "$audio" "$position"
      echo 0 > $out # We need to have an output
    '';

  # Position of the source during the recording.
  # The receiver is needed because the position is sampled at the sample frequency
  # of the recording at that specific receiver position.
#   get_position = { event, receiver, start, stop }:
#     (get_recording_and_position {inherit event receiver start stop;}).position;
#
#   # Immission recording of an event at a certain receiver.
#   get_recording = { event, receiver, start, stop }:
#     (get_recording_and_position {inherit event receiver start stop;}).recording;


  # Backpropagate an event recorded at a certain receiver.
  get_emission = {event, receiver, start, stop, settings_backpropagation ? {} }:
    let
      immission = get_immission {inherit event receiver start stop;};
      name = "${event}-${receiver}-emission";
      settings_in = builtins.toFile "settings_backpropagation.json" (builtins.toJSON settings_backpropagation);
      env = python.withPackages(ps: with ps; [ reverter ]);
    in runCommand name {
      buildInputs = [ env ];
      outputs = ["out" "audio" "position" "settings"];
      passthru.immission = immission;
    } ''
      python -m reverter ${event} ${receiver} ${immission.audio} ${immission.position} ${settings_in} $audio $position $settings
      echo 0 > $out # We need to have an output
    '';

  # Extract features from backpropagated recording.
  get_features = { event, receiver, start, stop, settings_backpropagation ? {}, settings_features ? {} }:
    let
      name = "${event}-${receiver}-features";
      env = python.withPackages(ps: with ps; [ features ]);
      emission = get_emission {inherit event receiver start stop settings_backpropagation;};
      source_position = emission.immission.position;
      settings_in = builtins.toFile "settings_features.json" (builtins.toJSON settings_features);
    in runCommand name {
      buildInputs = [ env ];
      outputs = ["out" "noise" "tones" "settings"];
      passthru.emission = emission;
    } ''
      python -m features ${receiver} ${emission.audio} ${source_position} ${settings_in} $noise $tones $settings
      echo 0 > $out # We need to have an output
    '';

  # Synthesise emission.
  get_synthesis = { event, receiver, start, stop, settings_backpropagation ? {}, settings_features ? {}, settings_synthesis ? {} }:
    let
      features = get_features {inherit event receiver start stop settings_backpropagation settings_features;};
      name = "${event}-${receiver}-synthesis";
      position = features.emission.position; # Synthesis of signal covers same interval as backpropagated signal.
      settings_in = builtins.toFile "settings_synthesis.json" (builtins.toJSON settings_synthesis);
      env = python.withPackages(ps: with ps; [ synthesis ]);
    in runCommand name {
      buildInputs = [ env ];
      outputs = ["out" "audio" "position" "settings"];
      passthru.features = features;
    } ''
      python -m synthesis ${features.noise} ${features.tones} ${features.settings} ${settings_in} ${position} $settings $audio $position
      echo 0 > $out # We need to have an output
    '';

  # Auralise an event
  # The settings are sets which are eventually passed as JSON.
  get_auralisation = { event, receiver, start, stop
                     , settings_backpropagation ? {}
                     , settings_features ? {}
                     , settings_synthesis ? {}
                     , settings_auralisation ? {}
                     }:
    let
      synthesis = get_synthesis {inherit event receiver start stop settings_backpropagation settings_features settings_synthesis;};
      # Write settings of auralisation to file
      name = "${event}-${receiver}-auralisation";
      env = python.withPackages(ps: with ps; [ auralisation ]);
      settings_in = builtins.toFile "settings_auralisation.json" (builtins.toJSON settings_auralisation);
    in runCommand name {
      buildInputs = [ env ];
      outputs = ["out" "audio" "position" "settings"];
      passthru.synthesis = synthesis;
    } ''
      python -m auralisation ${event} ${receiver} ${settings_in} ${synthesis.position} ${synthesis.audio} $position $audio $settings
      echo 0 > $out # We need to have an output
    '';

  # Convert from hdf to wav
  to_wav = { audio }:
    let
      env = python.withPackages(ps: with ps; [ acoustics h5store ]);
      name = (lib.removeSuffix ".hdf5" audio.name ) + ".wav";
    in runCommand name { buildInputs = [ env ]; } ''
      python -c "
from acoustics import Signal
from h5store import h5load

data, meta = h5load('${audio}')
signal = Signal(data, meta['fs']).normalize().to_wav('$out')
"
    '';

}



