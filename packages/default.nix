{ python35Packages
, fetchFromGitHub
, fetchurl
, runCommand
, database
, audio
}:

with python35Packages;

let

  #pythonModule = file: { ... } @attrs:
  #  let
  #    basename = (builtins.baseNameOf (builtins.toString file));
  #  in runCommand (python.libPrefix + "-" + basename) attrs ''
  #    mkdir -p $out/${python.sitePackages}
  #    cp ${file} $out/${python.sitePackages}/${basename}
  #  '';

  # Function that copies a single Python module into site-packages.
  pythonModule = { src, ... } @attrs:
    let
      filename = (builtins.baseNameOf (builtins.toString src));
    in mkPythonDerivation ({
      inherit src;
      name = filename;
      # We copy the file so we can modify it if we have to.
      unpackPhase = ''
        cp $src ${filename}
      '';
      installPhase = ''
        runHook preInstall
        mkdir -p "$out/${python.sitePackages}"
        cp *.py "$out/${python.sitePackages}/${filename}"
        runHook postInstall
      '';
    } // attrs);

in rec {
  acoustics = buildPythonPackage rec {
    pname = "acoustics";
    version = "1cbd530d7a592a6bc8114a8d1ceea17fdf6a0ff1";
    name = "${pname}-${version}";

    buildInputs = [ cython pytest ];
    propagatedBuildInputs = [ numpy scipy matplotlib tkinter pandas tabulate ];

    src = fetchFromGitHub {
      owner = "python-acoustics";
      repo = "python-acoustics";
      rev = version;
      sha256 = "0nmw6v90bzjv4ll12akph3vgrl07chms2msrkj3z9l7gi6krmw5l";
    };

    # Tests not distributed
    doCheck = false;
  };

  auraliser = buildPythonPackage rec {
    pname = "auraliser";
    version = "45cf5ca52d58cb2da18eeec99b633e668c1f8560";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = pname;
      rev = version;
      sha256 = "0s67m4gdddlpzy424151zfvf0x8c66sdq3lyf0gml8x3abrfp9yc";
    };

    doCheck = false;

    buildInputs = [ pytest ];
    propagatedBuildInputs = [ acoustics cytoolz geometry ism numpy scipy matplotlib numba scintillations streaming turbulence ];
  };

  geometry = buildPythonPackage rec {
    pname = "geometry";
    version = "9d4d38d003299d6655642143723f30339cf0a8ad";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = "python-geometry";
      rev = version;
      sha256 = "0qi7cg4kpkw3yqa7dn9sm1wi92a4921nza530db61cjv6yjimb4b";
    };

    buildInputs = [ pytest cython ];
    propagatedBuildInputs = [ numpy ];

    meta = {
      description = "Geometry module for Python";
    };
  };

  ism = buildPythonPackage rec {
    pname = "ism";
    version = "7790e80132d76273816862a0412513799939b02b";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = pname;
      rev = version;
      sha256 = "0qzdrk3aky1h1y3cbjy20pwg3pa7jdv6q0zp6377940frd7d9ngm";
    };

    doCheck = false;

    buildInputs = [ cython pytest ];
    propagatedBuildInputs = [ geometry numpy matplotlib cytoolz];
  };

  noisy  = buildPythonPackage rec {
    pname = "noisy";
    version = "57ec1efe3312f288583e351daffe3f3cef96d2c5";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = pname;
      rev = version;
      sha256 = "1fga85w87d0970h5zg3cjd6k4vqh994igp59fv7jzm93b2my0hc3";
    };

    doCheck = false;

    propagatedBuildInputs = [ numpy ];
  };

  streaming = buildPythonPackage rec {
    pname = "streaming";
    version = "ca602c6cd9f66786d56e0dbffd96b8a441d87f24";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = pname;
      rev = version;
      sha256 = "0r3gf9wzr5pfk89z8n3xfkd9c67w57mxnqrj7ynxzd6di0za6ncr";
    };

    doCheck = false;

    buildInputs = [ pytest cython ];
    propagatedBuildInputs = [ cytoolz multipledispatch numpy noisy scipy ];
  };

  scintillations = buildPythonPackage rec {
    pname = "scintillations";
    version = "4d0b321a09caaf890678b5212e377f549a9fe533";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = pname;
      rev = version;
      sha256 = "01y504gb339ihwp6gxpd1rj19vrfqdbkvn4bvhk5lv399l0nac81";
    };

    doCheck = false;

    buildInputs = [ pytest cython ];
    propagatedBuildInputs = [ acoustics numpy scipy streaming ];
  };

  turbulence = buildPythonPackage rec {
    pname = "turbulence";
    version = "9aee68f73b489b1cfe6263bcab5b819a7d3aaf90";
    name = "${pname}-${version}";

    src = fetchFromGitHub {
      owner = "FRidh";
      repo = pname;
      rev = version;
      sha256 = "1f0xa8l16c6z5igkcg3pcn3vhvxf39q654b6spl7rdzadcrgriwv";
    };

    doCheck = false;

    buildInputs = [ pytest ];
    propagatedBuildInputs = [ numpy scipy matplotlib pandas numexpr traitlets numtraits ];

  };

  # Modules

  auralisation = pythonModule {
    src = ./auralisation.py;
    propagatedBuildInputs = [ auraliser h5store sonorus ];
  };

  features = pythonModule {
    src = ./features.py;
    propagatedBuildInputs = [ acoustics sonorus pandas h5store ];
  };

  h5store = pythonModule {
    src = ./h5store.py;
    propagatedBuildInputs = [ pandas ];
  };

  recording = pythonModule {
    src = ./recording.py;
    propagatedBuildInputs = [ blaze h5store sonorus ];
  };

  reverter = pythonModule {
    src = ./reverter.py;
    propagatedBuildInputs = [ h5store sonorus auraliser ];
  };

  sonorus = pythonModule {
    inherit audio database;
    src = ./sonorus.py;
    propagatedBuildInputs = [ acoustics odo pandas blaze ];
  };

  synthesis = pythonModule {
    src = ./synthesis.py;
    propagatedBuildInputs = [ sonorus pandas h5store ];
  };
}
