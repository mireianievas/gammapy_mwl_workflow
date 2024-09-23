import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from regions import Regions

from gammapy.utils.scripts import make_path, make_name
from gammapy.maps import RegionNDMap, MapAxis, RegionGeom, WcsGeom
from gammapy.irf import EDispKernel, EDispKernelMap
from .ogip_spectrum_dataset import StandardOGIPDataset
from gammapy.data import GTI

__all__ = ["StandardOGIPDatasetReader"]


@classmethod
def from_hdulist(cls, hdulist, hdu1="MATRIX", hdu2="EBOUNDS"):
    """Create `EnergyDispersion` object from `~astropy.io.fits.HDUList`.

    Parameters
    ----------
    hdulist : `~astropy.io.fits.HDUList`
        HDU list with ``MATRIX`` and ``EBOUNDS`` extensions.
    hdu1 : str, optional
        HDU containing the energy dispersion matrix, default: MATRIX
    hdu2 : str, optional
        HDU containing the energy axis information, default, EBOUNDS
    """
    matrix_hdu = hdulist[hdu1]
    ebounds_hdu = hdulist[hdu2]

    data = matrix_hdu.data
    header = matrix_hdu.header

    pdf_matrix = np.zeros([len(data), header["DETCHANS"]], dtype=np.float64)

    for i, l in enumerate(data):
        if l.field("N_GRP"):
            m_start = 0

            for k in range(l.field("N_GRP")):
                if np.isscalar(l.field("N_CHAN")):
                    f_chan = l.field("F_CHAN")
                    n_chan = l.field("N_CHAN")
                else:
                    f_chan = l.field("F_CHAN")[k]
                    n_chan = l.field("N_CHAN")[k]

                try:
                    pdf_matrix[i, f_chan : f_chan + n_chan] = l.field("MATRIX")[
                        m_start : m_start + n_chan
                    ]
                except IndexError:
                    # This is probably a single-channel (e.g. UVOT uvot2pha rsp file)
                    pdf_matrix[i, f_chan : f_chan + n_chan] = np.asarray([l.field("MATRIX"),])[
                        m_start : m_start + n_chan
                    ]

                m_start += n_chan

    table = Table.read(ebounds_hdu)
    energy_min = table["E_MIN"].quantity
    energy_max = table["E_MAX"].quantity
    # in uvot2pha the energies are reversed in the EBOUNDS HDU:
    if energy_min[0]>energy_max[0]:
        energy_min,energy_max = energy_max,energy_min

    energy_edges = np.append(energy_min.value, energy_max.value[-1]) * energy_min.unit
    energy_axis = MapAxis.from_edges(energy_edges, name="energy", interp="lin")

    table = Table.read(matrix_hdu)
    energy_min = table["ENERG_LO"].quantity
    energy_max = table["ENERG_HI"].quantity
    
    # To avoid that min edge is 0
    energy_min[0] += 1e-2 * (energy_max[0] - energy_min[0])
    energy_edges = np.append(energy_min.value, energy_max.value[-1]) * energy_min.unit
    energy_true_axis = MapAxis.from_edges(
        energy_edges, name="energy_true", interp="lin"
    )

    return cls(axes=[energy_true_axis, energy_axis], data=pdf_matrix)


EDispKernel.from_hdulist = from_hdulist


class StandardOGIPDatasetReader:
    """Read `StandardOGIPDataset` from regular OGIP files.

    BKG file, ARF, and RMF can be set in the PHA header or be passed
    as arguments in the read method.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        OGIP PHA file to read
    """

    tag = "ogip"

    def __init__(self, filename, region_hdu="REGION", gti_hdu="GTI"):
        self.filename = make_path(filename)
        self.region_hdu = region_hdu
        self.gti_hdu = gti_hdu

    def get_valid_path(self, filename):
        """Get absolute or relative path

        The relative path is with respect to the name of the reference file.

        Parameters
        ----------
        filename : str or `Path`
            Filename

        Returns
        -------
        filename : `Path`
            Valid path
        """
        filename = make_path(filename)

        if not filename.exists():
            return self.filename.parent / filename
        else:
            return filename

    def get_filenames(self, pha_meta):
        """Get filenames

        Parameters
        ----------
        pha_meta : dict
            Meta data from the PHA file

        Returns
        -------
        filenames : dict
            Dict with filenames of "arffile", "rmffile" (optional)
            and "bkgfile" (optional)
        """
        filenames = {"arffile": self.get_valid_path(pha_meta["ANCRFILE"])}

        if "BACKFILE" in pha_meta:
            filenames["bkgfile"] = self.get_valid_path(pha_meta["BACKFILE"])

        if "RESPFILE" in pha_meta:
            filenames["rmffile"] = self.get_valid_path(pha_meta["RESPFILE"])

        return filenames

    def _read_regions(self, hdulist):
        """Read region data from an HDUlist."""
        region, wcs = None, None
        if self.region_hdu in hdulist:
            region_table = Table.read(hdulist[self.region_hdu])
            pix_region = Regions.parse(region_table, format="fits")
            pix_region = pix_region.shapes.to_regions()
            wcs = WcsGeom.from_header(region_table.meta).wcs

            regions = []
            for reg in pix_region:
                regions.append(reg.to_sky(wcs))
            region = list_to_compound_region(regions)

        return region, wcs

    def _read_gti(self, hdulist):
        """Read GTI table from input HDUList"""
        from astropy.time import Time
        gti = None
        if self.gti_hdu in hdulist:
            header = hdulist[self.gti_hdu].header
            gti_table = Table.read(hdulist[self.gti_hdu])
            ref_mjd = header['MJDREFI']+header['MJDREFF']
            if gti_table['START'].unit == 's':
                gti_table['START'] = Time(ref_mjd+gti_table['START'].to('s').value/86400.,format='mjd')
                gti_table['STOP']  = Time(ref_mjd+gti_table['STOP'].to('s').value/86400.,format='mjd')
            gti = GTI(gti_table)
        return gti

    @staticmethod
    def extract_spectrum(pha_table):
        """extract spectrum data from PHA table.

        The input table must follow OGIP format.
        Only Counts spectra are supported (not count rate).

        Currently the columns AREASCAL, STAT_ERR, SYS_ERR are not
        taken into account.

        The resulting dataset is rebinned according to the GROUPING
        column.
        """
        spectrum_data = {}
        pha_meta = pha_table.meta

        if pha_meta["HDUCLASS"] != "OGIP":
            raise ValueError("Input file is not an OGIP file.")
        if pha_meta["HDUCLAS1"] != "SPECTRUM":
            raise ValueError("Input file is not a PHA file.")
        if pha_meta["HDUCLAS2"] == "NET":
            raise ValueError("Subtracted PHA files are not supported.")
        if pha_meta["HDUCLAS3"] == "RATE":
            raise ValueError("Rate PHA files are not supported.")

        if "HDUCLAS4" in pha_meta:
            if pha_meta["HDUCLAS4"] == "TYPE:II":
                raise ValueError("Type II PHA files are not supported.")

        spectrum_data["livetime"] = pha_meta["EXPOSURE"] * u.s

        spectrum_data["channel"] = pha_table["CHANNEL"]
        spectrum_data["counts"] = pha_table["COUNTS"]

        mask_safe = True
        if "QUALITY" in pha_table.columns:
            mask_safe = pha_table["QUALITY"].data == 0
        spectrum_data["mask_safe"] = mask_safe

        grouping = None
        if "GROUPING" in pha_table.columns:
            grouping = pha_table["GROUPING"]
        spectrum_data["grouping"] = grouping

        if "BACKSCAL" in pha_table.columns:
            acceptance = pha_table["BACKSCAL"]
        else:
            acceptance = pha_meta["BACKSCAL"]
        spectrum_data["acceptance"] = acceptance

        exposure = pha_meta["EXPOSURE"]
        spectrum_data["acceptance"] *= exposure

        area_scale = 1
        if "AREASCAL" in pha_table.columns:
            area_scale = pha_table["AREASCAL"]
        spectrum_data["area_scale"] = area_scale

        return spectrum_data

    def read(self, filenames=None, name=None):
        hdulist = fits.open(self.filename, memmap=False)
        pha_table = Table.read(hdulist["spectrum"])

        data = self.extract_spectrum(pha_table)
        region, wcs = self._read_regions(hdulist)
        gti = self._read_gti(hdulist)

        if filenames is None:
            filenames = self.get_filenames(pha_meta=pha_table.meta)

        edisp_kernel = EDispKernel.read(filenames["rmffile"])
        energy_axis = edisp_kernel.axes["energy"]
        energy_true_axis = edisp_kernel.axes["energy_true"]

        # Some instruments, e.g. UVOT and BAT, do not have separate rmffile and arffile, but a respfile.
        if str(filenames["arffile"]).split("/")[-1] != "NONE":
            arf_table = Table.read(filenames["arffile"], hdu="SPECRESP")
        
        bkg_table = Table.read(filenames["bkgfile"])
        data_bkg = self.extract_spectrum(bkg_table)

        geom = RegionGeom(region=region, wcs=wcs, axes=[energy_axis])

        counts = RegionNDMap(geom=geom, data=data["counts"].data, unit="")
        acceptance = RegionNDMap(geom=geom, data=data["acceptance"], unit="")
        mask_safe = RegionNDMap(geom=geom, data=data["mask_safe"], unit="")

        counts_off = RegionNDMap(geom=geom, data=data_bkg["counts"].data, unit="")
        acceptance_off = RegionNDMap(geom=geom, data=data_bkg["acceptance"], unit="")

        geom_true = RegionGeom(region=region, wcs=wcs, axes=[energy_true_axis])

        if str(filenames["arffile"]).split("/")[-1] != "NONE":
            # standard case, arf_table contains the response
            exposure = arf_table["SPECRESP"].quantity * data["livetime"]
        else:
            # extract exposure and migration matrix from the rsp file. 
            exposure = edisp_kernel.data *u.Unit("cm2") * data["livetime"]
            #edisp_kernel.data[edisp_kernel.data>0] *= 1./edisp_kernel.data[edisp_kernel.data>0] 
            edisp_kernel.data = 1

        exposure = RegionNDMap(geom=geom_true, data=exposure.value, unit=exposure.unit)
        edisp = EDispKernelMap.from_edisp_kernel(edisp_kernel, geom=exposure.geom)

        index = np.where(data["grouping"] == 1)[0]
        if len(index)!=0:
            edges = np.append(energy_axis.edges[index], energy_axis.edges[-1])
            grouping_axis = MapAxis.from_energy_edges(edges, interp=energy_axis._interp)
        else:
            # no grouping to be applied to the data
            grouping_axis = energy_axis 

        name = make_name(name)
        dataset = StandardOGIPDataset(
            name=name,
            counts=counts,
            acceptance=acceptance,
            counts_off=counts_off,
            acceptance_off=acceptance_off,
            edisp=edisp,
            exposure=exposure,
            mask_safe=mask_safe,
            grouping_axis=grouping_axis,
            gti=gti,
            meta_table=pha_table.meta,
        )

        return dataset
