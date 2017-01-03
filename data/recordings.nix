{requireFile, writeTextFile, lib, runCommand, python3}:

let
  hashes = (builtins.fromJSON ( builtins.readFile ./recordings/recordings.json ));
  require = name: sha256: requireFile {inherit name sha256; message="Recording ${name} with sha256 ${sha256} is required.";};
  files =  (lib.mapAttrs require hashes);
  listWithFiles = (writeTextFile {name="recordings.json"; text=builtins.toJSON files;});
in runCommand "audio.hdf5" {buildInputs = [(python3.withPackages(ps: [ ps.h5py ]))];} ''
  python3 -c "
import os
import h5py
import json

with open('${listWithFiles}') as f:
    files = json.load(f)
#names = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], files))

with h5py.File('$out', 'w') as f:
    grp = f.create_group('recordings')
    for filename, path in files.items():
        name = os.path.splitext(filename)[0]
        grp[name] = h5py.ExternalLink(path, '/recordings/' + name)
"
''
