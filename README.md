# DNA-Aptamers

## Development

### Mac/Linux users

1. Get yourself Python 3.9

2. Setup the venv

```
python -m venv venv
```

3. Activate the venv

Bash shell (most of you guys)
```
source ./venv/bin/activate
```

Fish shell
```
. ./venv/bin/activate.fish
```

4. Install the python package `nupack`
```
cd libs

pip install [your-distro]
```

where `[your-distro]` is one of
```
 nupack-4.0.0.27-cp39-cp39-macosx_10_13_x86_64.whl
 nupack-4.0.0.27-cp39-cp39-macosx_11_0_arm64.whl
 nupack-4.0.0.27-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### Windows users

https://docs.nupack.org/start/#windows-installation

1. Get WSL2
2. Install Ubuntu from the Microsoft Store
3. Open Ubuntu
4. Follow the installation instructions for Mac/Linux
