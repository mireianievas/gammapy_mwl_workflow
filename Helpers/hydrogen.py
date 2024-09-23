import os
import sys
import requests
import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits

def parse_response(html_response):
    """ parses the html response and extracts the weighted total NH value """
    akey = "headers='htotw'>"
    a = html_response.index(akey) + len(akey)
    l = html_response[a:].index("</td>")
    part = html_response[a:a+l]
    if part.endswith('</sup>'):
        part = part[:-len('</sup>')]
        base, expo = part.split(' &times;10<sup>')
        return float(base) * 10**float(expo)
    else:
        return float(part)

def get_gal_nh_from_radec(ra, dec):
    url = "https://www.swift.ac.uk/analysis/nhtot/donhtot.php"
    """ asks swift.ac.uk for the total NH value at (ra,dec) """
    r = requests.post(url, data=dict(
        equinox=2000, Coords="%s %s" % (ra, dec),
        submit='Calculate NH')
        )
    nh = parse_response(r.text)
    return nh


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

def get_gal_nh(infile=None,srcname=None,src=None):
    try:
        ra,dec = get_ra_dec(infile,srcname,src)
        return get_gal_nh_from_radec(ra,dec)
    except Exception as e: 
        print(e)
        return(None)

def sherpa_xtbabs_model(infile=None,srcname=None,src=None):
    #os.environ['HEADAS'] = "/home/mnievas/Software/heasoft-6.32.1/INSTALL/x86_64-pc-linux-gnu-libc2.35"
    from sherpa.astro.xspec import XSTBabs
    from gammapy_ogip.models import SherpaSpectralModel
    
    abs_model = XSTBabs()
    nhgal = get_gal_nh(infile,srcname,src)/1e22
    abs_model.nh = nhgal
    abs_model.nh.frozen = True
    abs_model = SherpaSpectralModel(abs_model, default_units=(u.keV, 1))
    abs_model.tag = "sherpa.astro.xspec.XSTBabs"
    return(abs_model)

def generate_tbabs_interp_table(outfile):
    #os.environ['HEADAS'] = "/home/mnievas/Software/heasoft-6.32.1/INSTALL/x86_64-pc-linux-gnu-libc2.35"
    from sherpa.astro.xspec import XSTBabs
    from astropy.io import fits as pyfits
    from astropy.coordinates import SkyCoord
    from gammapy_ogip.models import SherpaSpectralModel
    
    nH_array = np.round(np.linspace(-4,2.5,10*60),decimals=5)
    en_array = np.round(np.linspace(-2,2.5,10*10),decimals=5)
    
    tbabs_models = {
        nH: XSTBabs() for nH in nH_array
    }
    
    tbabs_table = [] #names = en_array)
    
    for k,nH in enumerate(tbabs_models):
        tbabs_models[nH].nh.val = 10**nH
        tbabs_models[nH] = SherpaSpectralModel(tbabs_models[nH],default_units=(u.keV, 1))
        tau = -np.log(tbabs_models[nH](10**en_array *u.keV))
        tau[tau==np.inf] = 100
        tau[tau<0] = 0
        tbabs_table += [list(np.round(tau,5))]

    tbabs_table = Table(data=tbabs_table)
    tbabs_table.meta['log10_E_type']  = 'rows'
    tbabs_table.meta['log10_E_values']  = list(en_array)
    tbabs_table.meta['log10_nH_type'] = 'columns'
    tbabs_table.meta['log10_nH_values'] = list(nH_array)
    tbabs_table.meta['table data'] = 'tau factor (natural log of absorption)'

    tbabs_table.write(outfile,format="ascii.ecsv",overwrite=True)


def get_tbabs_template_model(tbabsfile,infile=None,srcname=None,src=None,freeze=True):
    from gammapy.modeling.models import TemplateNDSpectralModel
    from gammapy.maps import RegionNDMap, MapAxis
    
    # add the hydrogen absorption from the 2D table (output from sherpa's / xspec's tbabs). It is the same for XRT.
    tbabs_table = Table.read(tbabsfile,format='ascii.ecsv')
    tbabs_data  = tbabs_table.as_array()
    tbabs_data  = tbabs_data.view(np.float64).reshape(tbabs_data.shape + (-1,))
    
    log_nh_array_20 = np.asarray(tbabs_table.meta['log10_nH_values'])
    log_en_array    = np.asarray(tbabs_table.meta['log10_E_values'])
    
    energy_axis = MapAxis(nodes=10**log_en_array,node_type='center',interp='log',name='energy_true',unit='keV')
    nh_axis     = MapAxis(nodes=10**log_nh_array_20*1e22,node_type='center',interp='log',name='nH',unit='cm-2')

    # avoid possible issues related to zeros in the likelihood.
    transmission = np.exp(-np.transpose(tbabs_data))
    transmission[transmission<1e-5] = 1e-5
    
    region_ndmap = RegionNDMap.create(
        region = None,
        axes = [energy_axis,nh_axis],
        data = transmission,
    )
    template_abs_model = TemplateNDSpectralModel(
        map = region_ndmap, 
        interp_kwargs={'method': 'linear', 'fill_value': 1e-5}
    )

    nhgal = get_gal_nh(infile,srcname,src)
    if nhgal is not None:
        template_abs_model.nH.quantity = nhgal*u.Unit('cm-2')
        if freeze:
            template_abs_model.nH.frozen = True
    else:
        print('Generating generic template absorption model with a default nH')
        
    return(template_abs_model)