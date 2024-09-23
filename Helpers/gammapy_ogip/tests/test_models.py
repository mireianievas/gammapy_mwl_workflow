import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u

from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

from sherpa.astro.xspec import XSwabs
from sherpa.models import PowLaw1D
from ..models import SherpaSpectralModel


def test_SherpaSpectralModel():
    energy_grid = np.linspace(0.5, 10.0, 10) * u.keV

    plaw = PowLaw1D()
    plaw.ampl = 1e-3
    plaw.gamma = 2

    abs_model = XSwabs()
    abs_model.nH = 5

    # Gammapy wrapper
    f1 = SherpaSpectralModel(plaw)
    f2 = SherpaSpectralModel(abs_model, default_units=(u.keV, 1))
    f3 = f1 * f2

    # Plain sherpa
    plaw_with_abs = plaw * abs_model

    assert_allclose(f3(energy_grid).value[:-1], plaw_with_abs(energy_grid.value)[:-1])
    SkyModel(spectral_model=f3)  # Test evaluate on simple geom
    with pytest.raises(AttributeError):
        SkyModel(spectral_model=f2)  # Wrong units, f2 is an absorption model
