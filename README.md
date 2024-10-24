Repository containing multi-wavelength (MWL) data of the distant quasar OP 313 (z=0.997) taken during the nights of MJD 60373 (3-4 March 2024) and MJD 60384 (14-15 March 2024) using _Fermi_-LAT, _Swift_-XRT, _Swift_-UVOT, and the Liverpool IO:O photometric data.

### Directory structure:

* Notebooks/DatasetGenerator: Notebooks (one per instrument) that summarize the steps to generate gammapy-compliant 1D and 3D binned datasets.
* Notebooks/DatasetAnalysis: Notebooks (one per instrument) to analyse each dataset independently and one notebook to perform the MWL joint analysis.
* Helpers: Auxiliary functions to generate native multiplicative models for dust extinction (out of xspec's redden), neutral hydrogen (out of xspec's tbabs), EBL absorption, and utility functions for file handling and plotting.
* Models: multiplicative models in tabular format for easy 2d interpolation.
* Figures: collection of figures for the paper.

Based on the original work from https://github.com/luca-giunti/gammapyXray

Zenodo: https://zenodo.org/records/13837637 
