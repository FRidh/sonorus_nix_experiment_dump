{runCommand, python, get_event}:

rec {

  # Cut and fade a WAVE file.
  cut_and_fade = { stimuli, cut_start ? 5, cut_end ? 5, fade_start ? 2, fade_end ? 2 }:
    let
      name = stimuli.name + '-faded.wav';
      env = python.withPackages(ps: with ps; [ acoustics ]);
      script = ./cut_and_fade.py;
    in runCommand name {
      buildInputs = [ env ];
    } ''
      python ${script} ${cut_start} ${cut_end} ${fade_start} ${fade_end} ${stimuli} $out
    '';


  # Function to create a listening test.
  # We pass our requirements, and then run a Python script to choose events.
  # Auralisations are created of the chosen events.
  # These are passed to another script that generates an HTML file.
  # This is the final result of this function.
  create_listening_test = { nevents                 # Amount of events
                          , aircraft ? [ "A320" ]   # List of aircraft types
#                           , simulation_settings     # Set with simulation settings sets
                          }:
    let
      name = "listening-test.html";
      env = python.withPackages(ps: with ps; [ ]);
      generate_test = ./generate_test.py;

      # Create a selection of events. Import from derivation!
      event_names = builtins.fromJSON builtins.realFile (let
          env = python.withPackages(ps: with ps; [ sonorus ]);
          name = "event-names.json";
        in runCommand {
          buildInputs = [ env ];
        } ''
          python -c "

          "
        '');

      events = map ( get_event event_names );
      recordings = map
      events_file = builtins.toFile (builtins.toJSON "events.json");


    in runCommand name {
      buildInputs = [ env ];
      passthru.events = events;
    } ''
      python -m ${generate_test} ${events_file} $out
    '';
}
