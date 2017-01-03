with import <nixpkgs> {};
let
  hash = "7cd1a58a753e0cf7bb17dfc96ad51d3dbd3a354a";
in with import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/${hash}.tar.gz") {};

let
  event = "12_105_A320";
in rec {
  # Configuration file
  #config = builtins.fromJSON (builtins.readFile ./config.json);

  packages = callPackage ./packages {inherit (data) database audio;};

  # Measurement data
  data = import ./data {inherit (pkgs) requireFile runCommand writeTextFile; inherit lib python3;};

  # Functions for analysing individual events
  _event = import ./events {inherit lib; inherit (pkgs) runCommand; inherit python;};

  #recordings = callPackage ./recordings {};

  python = let
    overrides = self: super: packages;
  in (python35.override{packageOverrides=overrides;});

  # Environment with notebook for development
  env = python.withPackages(ps: with ps; [notebook auraliser blaze h5store seaborn sonorus]);

  settings_backpropagation = {}; #builtins.fromJSON (builtins.readFile ./settings/reverter.json);
  settings_features = {};
  settings_synthesis = {}; #builtins.fromJSON (builtins.readFile ./settings/synthesis.json);
  settings_auralisation = {}; #builtins.fromJSON (builtins.readFile ./settings/auralisation.json);


  test_immission = _event.to_wav { audio = (_event.get_immission {
    event=event;
    receiver="A";
    start="-10";
    stop="+10";
  }).audio;};

  test_reverted = _event.to_wav { audio = (_event.get_emission {
    event=event;
    receiver="A";
    start="-10";
    stop="+10";
    inherit settings_backpropagation;
  }).audio;};

  test_features = _event.get_features {
    event=event;
    receiver="A";
    start="-10";
    stop="+10";
    inherit settings_backpropagation settings_features;
  };

  test_synthesis = _event.to_wav { audio = (_event.get_synthesis {
    event=event;
    receiver="A";
    start="-10";
    stop="+10";
    inherit settings_backpropagation settings_features settings_synthesis;
  }).audio;};

  test_auralisation = _event.to_wav { audio = (_event.get_auralisation {
    event=event;
    receiver="A";
    start="-10";
    stop="+10";
    inherit settings_backpropagation settings_features settings_synthesis settings_auralisation;
  }).audio;};


  get_event = { event, receiver ? "A", start ? "-10", stop ? "+10"
              , settings_backpropagation ? {}, settings_features ? {}, settings_synthesis ? {}, settings_auralisation ? {} }:

              rec {
                immission = _event.get_immission {inherit event receiver start stop;};
                emission = _event.get_emission {inherit event receiver start stop settings_backpropagation;};
                features = _event.get_features {inherit event receiver start stop settings_backpropagation settings_features;};
                synthesis = _event.get_synthesis {inherit event receiver start stop settings_backpropagation settings_features settings_synthesis;};
                auralisation = _event.get_auralisation {inherit event receiver start stop settings_backpropagation settings_features settings_synthesis settings_auralisation;};
              };

  test_event = get_event { event=event;};
}
