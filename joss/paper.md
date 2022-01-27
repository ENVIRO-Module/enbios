---
title: 'ENBIOS: Environmental and Bioeconomic Assessment of Energy Transition Pathways '
tags:
  - Python
  - Energy Transition
  - Life CyCle Assessment
  - ...
authors:
  - name: Rafael Nebot 
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Nicholas Martin 
    orcid: 0000-0000-0000-0000
    affiliation: 2 
  - name: Laura Talens-Peiró 
    orcid: 0000-0002-1131-1838
    affiliation: 2
  - name: Cristina Madrid-López^[corresponding author]
    orcid: 0000-0002-4969-028X
    affiliation: 2

affiliations:
 - name: Institut de Ciència i Tecnologia Ambientals. Universitat Autònoma de Barcelona, Spain.
   index: 2
 - name: Departament d'Enyineria Quimica, Biologica i Ambiental. Universitat Autònoma de Barcelona, Spain.
   index: 3
 - name: Instituto Tecnológico de Canarias
   index: 1
date: 31 January 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Many of the long-term energy decisions being made for the “sustainable” energy transition are based on the results of energy models that do not include environmental constraints other than emissions. While essential for decision making, such assessments on their own offer incomplete and potentially misleading information about the sustainability of an energy pathway.  Combining system configuration data from such models with high resolution life cycle assessment information within a socio-ecological metabolism framework enables the complex relationships between environmental impacts, material constraints and bio-economic functions to be better interrogated. Here, we introduce ENBIOS, an environmental assessment framework created to support the energy transition by assessing a wider range of useful indicators. ENBIOS takes energy system optimization scenarios and connects life cycle assessment and social metabolism indicators for the multi-scale analysis of energy systems. The functionality of the module is demonstrated via a case study involving projected European energy system pathways for 2030 and 2050 modelled with Calliope. While significant drops in greenhouse gas emissions are projected under this scenario alongside rises in employment provision, land requirements and raw material supply risk are also predicted to rise significantly.

# Statement of need

`ENBIOS` is a Python package for for environmental assessment. 

RAFA PLEASE CHECK

Python enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`ENBIOS` is designed to be used by analysis with knowledge of life cycle assessment and MuSIASEM. The scientific publication explaining the methodological framework behind it has also bee published [@Martin:2022] a

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References