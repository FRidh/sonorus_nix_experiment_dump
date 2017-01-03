{lib, requireFile, writeTextFile}:

let
  hashes = (builtins.fromJSON ( builtins.readFile ./recordings.json ));
  require = name: sha256: requireFile {inherit name sha256; message="Recording ${name} with sha256 ${sha256} is required.";};
in rec {
  files =  (lib.mapAttrs require hashes);

  listWithFiles = (writeTextFile {name="recordings"; text=builtins.toJSON files;});

#  h = writeTextFile {name="json"; text=(builtins.toJSON hashes);};
#   x = files."A1002_1.WAV.hdf5";
}



