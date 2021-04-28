# EnBios-Enviro
Indicators of environmental sustainability of energy systems using MuSIASEM and LCA methodologies

<Intro>

**Disclaimer**: this README is still under elaboration, details may be missing or innacurate.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

<!-- Insert a representative Screenshot or an animated GIF (use "Recordit"?)-->

## Table of Contents

- [Documentation](#documentation)
- [Getting started](#getting-started)
  - [Features](#features)
  - [Installing and executing **nis-backend**](#installing-and-executing-nis-backend)
    - [Pip package](#pip-package)
    - [Docker image](#docker-image)
    - [Source code](#source-code)
- [Accessing **nis-backend** in Python and R scripts with **nis-client**](#accessing-nis-backend-in-python-and-r-scripts-with-nis-client)
- [People](#people)
- [Contact](#contact)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Keywords](#keywords)
  
## Getting started

### Features

- **High level methods to quickly obtain/refresh analyses**

### Installing and executing "enbios"

**enbios** is a Python package. It can be installed with the following two methods:

#### pip package

The pip version is obviously for execution as package.

* Set a Python environment
* Install the package with

`pip install enbios`

* Use class **ENVIRO**.

#### Source code

**NOTE**: Python3 and git required.

Clone this repository and execute "python setup.py install":

```
git clone https://github.com/ENVIRO-Module/enviro
cd enviro
git checkout develop
python3 setup.py install
```

## People

An enumeration of people who have contributed in different manners to the elaboration of NIS during MAGIC project lifetime:

* Rafael Nebot. ITC-DCCT (Instituto Tecnológico de Canarias - Departamento de Computación)
* Cristina Madrid. ICTA-UAB (Institut de Ciència i Tecnologia Ambientals - Universitat Autònoma de Barcelona)
* Laura Talens Peiró. ICTA-UAB.
* Nicholas Martin. ICTA-UAB.

## Contact

Please send any question regarding this repository to [rnebot@itccanarias.org](mailto:rnebot@itccanarias.org).

## License
This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details

## Acknowledgements
The development of this software was supported by the European Union’s Horizon 2020 research and innovation programme under 
Grant Agreement No. 837089 (Sentinel Energy). This work reflects the authors' view only; the funding agencies are not responsible for any use that may be made of the information it contains.

## Keywords

    Sustainability - Bioeconomy - Socio-Ecological Systems - Complex Adaptive Systems - Water-Energy-Food Nexus
