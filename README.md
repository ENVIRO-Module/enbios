# ENBIOS 
Environmental and bioeconomic system assessment for energy system optimization model (ESOM) scenarios.
<Intro>


[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

<!-- Insert a representative Screenshot or an animated GIF (use "Recordit"?) -->

## Table of Contents

  
## What is enbios
ENBIOS (Environmental and Bioeconomic System Analysis) is an assessment framework designed for the assessment of the **environmental impacts and resource use of energy pathways resulting from energy system optimization models (ESOMs)**.

It  integrates Life Cycle Assessment (LCA) and Social Metabolism Assessment using the Multi-Scale Integrated Assessment of Socio-Ecosystem Metabolism (MuSIASEM). It has been been co-designed with decision makers and energy modellers within the SENTINEL project (see acknowledgements).

The related python package [`enbios`](https://pypi.org/project/enbios/) takes data on energy system design and impact assessment methods to return a characterization matrix filled with bioeconomic and environmental indicators.  

More information on the roots of the framework can be found in [Deliverable 2.2]() 

### Data inputs
- Data on electricity production and power capacity structured by the [`friendly_data`](https://pypi.org/project/friendly-data/) package. 
- LCA inventories in .spold format
- Impact assessment methods in excel or csv format

### Outputs 
For each energy function and technology:
- Environmental impact indicators from the most used LCIA methods (Recipe2016, CML, AWARE, etc.)
- Metabolism indicators in rates of energy or materials per hour of human activity, power capacity and land use 
- Environmental externalization rates
- Raw Material Recycling rates and Supply risk


### Features

- Integration of LCA and MuSIASEM evaluation methods
- Import of .spold LCA inventory data to a multi-level tree-like setting
- Library of impact assessment methods based on LCIA
- New impact assessment methods developed for raw materials and circularity
- Consideration of externalized environmental impacts
- Takes data from the friendly-data package (other formats under development)
- High level methods to quickly obtain/refresh analyses


## Installing "enbios"

The python package can be installed with the following two methods:

### pip package

* Set a Python environment
* Install the package:

`pip install enbios`

### Source code

Clone Github repository and execute "python setup.py install":

```
git clone https://github.com/ENVIRO-Module/enbios
cd enviro
git checkout develop
python3 setup.py install
```

### Documentation

- The quick start guide can be found [here]()
- The user manual can be found [here]()

## People


* [Cristina Madrid-Lopez](https://portalrecerca.uab.cat/en/persons/cristina-madrid-lopez-3). - [ICTA-UAB](https://www.uab.cat/icta/) 
* [Nicholas Martin](https://portalrecerca.uab.cat/en/persons/nicholas-martin-4). - [ICTA-UAB](https://www.uab.cat/icta/)
* [Rafael Nebot](https://www.linkedin.com/in/rafael-j-nebot-medina-31551a12/). - [ITC-DCCT](https://www.itccanarias.org/web/en/)
* [Laura Talens-Peiro](https://portalrecerca.uab.cat/en/persons/laura-talens-peir%C3%B3-6).  -[ICTA-UAB](https://www.uab.cat/icta/) 

## Contact

- For questions about the enbios framework, please contact [cristina.madrid@uab.cat](mailto:cristina.madrid@uab.cat).
- For technical questions about the python package, please contact [rnebot@itccanarias.org](mailto:rnebot@itccanarias.org).

## License
<!-- This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details-->
<!-- Rafa, can we take a decision about the license??-->
## Acknowledgements
The development of ENBIOS is part of the work carried out by the [SosteniPra](https://www.sostenipra.cat/) Research group, at the Institute of Environmental Science and Technology of the Universitat Autonoma de Barcelona ([ICTA-UAB](https://www.uab.cat/icta/)) within work package 2 of the Horizon 2020 project Sustainable Energy Transitions Laboratory ([SENTINEL](https://sentinel.energy>), GA 837089).

The python package `enbios` is built in collaboration with the Technical Institute of the Canary Islands ([ITC](https://www.itccanarias.org/web/es/)) and based on the Nexus Information System developed within the Horizon 2020 project [MAGIC-nexus](https://magic-nexus.eu/) and the LCA-MuSIASEM integration protocol developed in the Marie Curie project [IANEX](https://cordis.europa.eu/project/id/623593).

This work reflects the authors' view only; the funding agencies are not responsible for any use that may be made of the information it contains.

## Keywords

    Energy Transitions - Sustainability - Bioeconomy - Complex Adaptive Systems - Circular Economy - Energy Modelling - 
