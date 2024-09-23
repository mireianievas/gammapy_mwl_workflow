from gammapy.modeling.models import (
    EBLAbsorptionNormSpectralModel
)
def get_ebl_absorption_component(z,model="saldana-lopez21"):
    files = {
        'saldana-lopez21': f"../Models/ebl_saldana-lopez_2021.fits.gz",
        'dominguez': f"../Models/ebl_dominguez11.fits.gz",
    }
    return EBLAbsorptionNormSpectralModel.read(
            files[model],
            redshift=z,
            alpha_norm=1,
        )
