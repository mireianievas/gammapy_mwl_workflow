import os
import sys
import requests
import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits

def parse_response(html_response):
    """ parses the html response and extracts the weighted total E(B-V) value
        Extinction law for SFD: Cardelli et al.(1989), O'Donnell(1994) 
        + Indebetouw et al. (2005) for IRAC, WISE.
    """
    import re
    match = re.search(
        r"<refPixelValueSFD>\s*([0-9.]+)\s*\(mag\)", 
        html_response
    )
    return (float(match.group(1)))
    
def get_gal_extinction_from_radec(ra, dec):
    url = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"
    url = f"{url}?locstr={ra}+{dec}+equ+j2000"
    """ asks IRSA IPAC Caltech for the total E(B-V) value at (ra,dec) """
    r = requests.get(url)
    ebv = parse_response(r.text)
    return ebv

def get_ra_dec(infile=None,srcname=None,src=None):
    if infile is not None:
        header = pyfits.getheader(infile)
        ra  = header['RA_OBJ']
        dec = header['DEC_OBJ']
    elif srcname is not None:
        src = SkyCoord.from_name(srcname)
        ra  = src.ra.deg
        dec = src.dec.deg
    elif src is not None:
        ra  = src.ra.deg
        dec = src.dec.deg
    else:
        raise(Exception("Need to specify either infile, srcname or a astropy SkyCorrd as src"))
        
    return(ra,dec)

def get_gal_extinction(infile=None,srcname=None,src=None):
    try:
        ra,dec = get_ra_dec(infile,srcname,src)
        return get_gal_extinction_from_radec(ra,dec)
    except Exception as e: 
        print(e)
        return(None)

def sherpa_xredden_model(infile=None,srcname=None,src=None):
    #os.environ['HEADAS'] = "/home/mnievas/Software/heasoft-6.32.1/INSTALL/x86_64-pc-linux-gnu-libc2.35"
    from sherpa.astro.xspec import XSredden
    from gammapy_ogip.models import SherpaSpectralModel

    # Calculate the values
    abs_model = XSredden()
    abs_model.E_BmV.val = get_gal_extinction(infile,srcname,src)
    abs_model.E_BmV.frozen = True
    abs_model = SherpaSpectralModel(abs_model,default_units=(u.keV, 1))
    abs_model.tag = "sherpa.astro.xspec.XSredden"
    return(abs_model)

def generate_xredden_interp_table(outfile):
    #os.environ['HEADAS'] = "/home/mnievas/Software/heasoft-6.32.1/INSTALL/x86_64-pc-linux-gnu-libc2.35"
    from sherpa.astro.xspec import XSredden
    from astropy.io import fits as pyfits
    from astropy.coordinates import SkyCoord
    from gammapy_ogip.models import SherpaSpectralModel
    
    ebv_array = np.round(np.linspace(0,2.5,10*25),decimals=5)
    en_array = np.round(np.linspace(-1,2,5*25),decimals=5)
    
    xredden_models = {
        ebv: XSredden() for ebv in ebv_array
    }
    
    xredden_table = []
    
    for k,ebv in enumerate(xredden_models):
        xredden_models[ebv].E_BmV.val = ebv
        xredden_models[ebv] = SherpaSpectralModel(xredden_models[ebv],default_units=(u.keV, 1))
        tau = -np.log(xredden_models[ebv](10**en_array *u.eV))
        tau[tau==np.inf] = 100
        tau[tau<0] = 0
        xredden_table += [list(np.round(tau,5))]
    
    xredden_table = Table(data=xredden_table)
    xredden_table.meta['log10_E_type']  = 'rows'
    xredden_table.meta['log10_E_values']  = list(en_array)
    xredden_table.meta['ebv_type'] = 'columns'
    xredden_table.meta['ebv_values'] = list(ebv_array)
    xredden_table.meta['table data'] = 'tau factor (natural log of absorption)'

    xredden_table.write(outfile,format="ascii.ecsv",overwrite=True)


def get_xredden_template_model(xreddenfile,infile=None,srcname=None,src=None,freeze=True):
    from gammapy.modeling.models import TemplateNDSpectralModel
    from gammapy.maps import RegionNDMap, MapAxis
    
    # add the extinction from the 2D table (output from sherpa's / xspec's redden). It is the same for XRT.
    xredden_table = Table.read(xreddenfile,format='ascii.ecsv')
    xredden_data = np.asarray([[k for k in j] for j in xredden_table])
    xredden_data[xredden_data>1e5] = 1e5
    ebv_array    = np.asarray(xredden_table.meta['ebv_values'])
    log_en_array = np.asarray(xredden_table.meta['log10_E_values'])
    
    energy_axis = MapAxis(nodes=10**log_en_array,node_type='center',interp='log',name='energy_true',unit='eV')
    ebv_axis    = MapAxis(nodes=ebv_array,node_type='center',interp='lin',name='ebv',unit='')

    # avoid possible issues related to zeros in the likelihood.
    transmission = np.exp(-np.transpose(xredden_data))
    transmission[transmission<1e-5] = 1e-5
    
    region_ndmap = RegionNDMap.create(
        region = None,
        axes = [energy_axis,ebv_axis],
        data = transmission,
    )
    
    template_abs_model = TemplateNDSpectralModel(
        map = region_ndmap,
        interp_kwargs={'method': 'linear', 'fill_value': 1e-5}
    )

    ebv = get_gal_extinction(infile,srcname,src)
    if ebv is not None:
        template_abs_model.ebv.quantity = ebv
        if freeze:
            template_abs_model.ebv.frozen = True
    else:
        print('Generating generic template absorption model with a default E(B-V)')
    
    return(template_abs_model)
