# emc2d

A Python3 implementation for EMC motion correction for low-dose TEM images.

## Install

```bash
$ git clone git@github.com:Varato/emc2d.git
$ cd emc2d
```

`pybind11` is added as a git submodule. To fetch it, run

```bash
$ git submodule init
$ git submodule update
```

Then

```buildoutcfg
$ python setup.py install
```

This will invoke cmake and build extensions.
