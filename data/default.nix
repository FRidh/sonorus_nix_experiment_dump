{requireFile, writeTextFile, lib, runCommand, python3}:

rec {

  # Single audio file
  #audio = requireFile rec {
  #  name = "audio.hdf5";
  #  sha256 = "9bdeb6aa77f9993e02d84ef1afbb80fcec88daecf6cfbd14e0069f35d1b401bb";
  #  message = "${name} with sha256 ${sha256} is required.";
  #};

  # Audio file
  audio = import ./recordings.nix {inherit requireFile writeTextFile lib runCommand python3;};

  # Database file
  database = requireFile rec {
    name = "sonair.sqlite3";
    sha256 = "dacb8daf5622bd3aa6374eebc0e0515ef61eb04484fb58b86f23fc5ebc13358d";
    message = "${name} with sha256 ${sha256} is required.";
  };
}
