# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.datasets import SpectrumDatasetOnOff
from astropy import units as u

__all__ = ["StandardOGIPDataset"]


class StandardOGIPDataset(SpectrumDatasetOnOff):
    """Dataset containing spectral data as defined by X-ray OGIP compliant files.

    A few elements are added that are not supported by the current SpectrumDataset in gammapy.

    grouping contains the information on the grouping scheme, namely the group number of each bin.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    counts_off : `~gammapy.maps.WcsNDMap`
        Ring-convolved counts cube
    acceptance : `~gammapy.maps.WcsNDMap`
        Acceptance from the IRFs
    acceptance_off : `~gammapy.maps.WcsNDMap`
        Acceptance off
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask_fit : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel. Unused here.
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask_safe : `~gammapy.maps.WcsNDMap`
        Mask defining the safe data range.
    grouping_axis : `~gammapy.maps.MapAxis`
        MapAxis defining the grouping scheme.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.
    name : str
        Name of the dataset.

    """

    stat_type = "wstat"
    tag = "StandardOGIPDataset"

    def __init__(self, *args, **kwargs):
        self._grouped = None
        axis = kwargs.pop("grouping_axis", None)
        super().__init__(*args, **kwargs)
        self.grouping_axis = axis
        if axis.nbin > 1:
            self.apply_grouping(self.grouping_axis)

    @property
    def grouped(self):
        """Return grouped SpectrumDatasetOnOff."""
        return self._grouped

    @property
    def _is_grouped(self):
        return self.grouped is not None

    def apply_grouping(self, axis=None):
        """Apply grouping."""
        if axis is None:
            raise ValueError("A grouping MapAxis must be provided.")
        else:
            dataset = self.to_spectrum_dataset_onoff()
            self._grouped = dataset.resample_energy_axis(
                axis, name=f"group_{self.name}"
            )

    @property
    def models(self):
        """Models (`~gammapy.modeling.models.Models`)."""
        return self.grouped._models

    @models.setter
    def models(self, models):
        """Models setter"""
        if self._is_grouped:
            self.grouped.models = models

    @property
    def mask_fit(self):
        """RegionNDMap providing the fitting energy range."""
        if self._is_grouped:
            return self.grouped.mask_fit

    @mask_fit.setter
    def mask_fit(self, mask_fit):
        """RegionNDMap providing the fitting energy range."""
        if self._is_grouped:
            self.grouped.mask_fit = mask_fit.resample_axis(
                axis=self.grouping_axis, ufunc=np.logical_or
            )

    @property
    def mask(self):
        """Combined fit and safe mask"""
        return self.grouped.mask

    def npred(self):
        """Predicted source and background counts
        Returns
        -------
        npred : `Map`
            Total predicted counts
        """
        return self.grouped.npred()

    def npred_signal(self, model_name=None):
        """ "Model predicted signal counts.
        If a model is passed, predicted counts from that component is returned.
        Else, the total signal counts are returned.
        Parameters
        -------------
        model_name: str
            Name of  SkyModel for which to compute the npred for.
            If none, the sum of all components (minus the background model)
            is returned
        Returns
        ----------
        npred_sig: `gammapy.maps.Map`
            Map of the predicted signal counts
        """
        return self.grouped.npred_signal(model_name=model_name)

    def stat_sum(self):
        """Total statistic given the current model parameters."""
        return self.grouped.stat_sum()

    def plot_fit(
        self,
        ax_spectrum=None,
        ax_residuals=None,
        kwargs_spectrum=None,
        kwargs_residuals=None,
    ):
        ax_spectrum, ax_residuals = self.grouped.plot_fit(
            ax_spectrum, ax_residuals, kwargs_spectrum, kwargs_residuals
        )

        return ax_spectrum, ax_residuals

    def plot_residuals_spectral(self, ax=None, method="diff", region=None, **kwargs):
        """Plot spectral residuals.
        The residuals are extracted from the provided region, and the normalization
        used for its computation can be controlled using the method parameter.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        method : {"diff", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals`.
        region: `~regions.SkyRegion` (required)
            Target sky region.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        return self.grouped.plot_residuals_spectral(
            ax=ax, method=method, region=region, **kwargs
        )

    @classmethod
    def read(cls, filename):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder.

        For formats specs see `OGIPDatasetReader.read`

        Parameters
        ----------
        filename : `~pathlib.Path` or str
            OGIP PHA file to read
        """
        from .io_ogip import StandardOGIPDatasetReader

        reader = StandardOGIPDatasetReader(filename=filename)
        return reader.read()

    def write(self, filename, overwrite=False, format="ogip"):
        raise NotImplementedError("Standard OGIP writing is not supported.")

    def to_spectrum_dataset_onoff(self, name=None):
        """convert to spectrum dataset on off by dropping the grouping axis.
        Parameters
        ----------
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `SpectrumDatasetOnOff`
            Spectrum dataset on off.
        """
        kwargs = {"name": name}

        kwargs["acceptance"] = self.acceptance
        kwargs["acceptance_off"] = self.acceptance_off
        kwargs["counts_off"] = self.counts_off
        dataset = self.to_spectrum_dataset()
        if "livetime" not in dataset.exposure.meta:
            if dataset.gti is not None:
                dataset.exposure.meta['livetime'] = dataset.gti.time_sum
            elif 'EXPOSURE' in dataset.meta_table:
                dataset.exposure.meta['livetime'] = dataset.meta_table['EXPOSURE']*u.s
                
        return SpectrumDatasetOnOff.from_spectrum_dataset(dataset=dataset, **kwargs)
