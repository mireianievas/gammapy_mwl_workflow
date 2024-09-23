import os,sys
import errno
import os.path
import glob
import numpy as np
import astropy.units as u

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

def rm_rf(path):
    try: 
        os.remove(path)
    except OSError: 
        pass

def get_dataset_energy_edges(dataset,b='center',factor=0.2):
    myexp = dataset.exposure
    exposure_data = myexp.data[:,0,0]
    more_than_factor = exposure_data > exposure_data.max()*factor
    good = myexp.geom.axes['energy_true'].center[more_than_factor]
    good_min = myexp.geom.axes['energy_true'].edges_min[more_than_factor]
    good_max = myexp.geom.axes['energy_true'].edges_max[more_than_factor]
    emin,emax = np.min(good_min),np.max(good_max)
    return(emin,emax)

def get_dataset_energy_edges_from_aeff_integral(dataset, b='center', factor=0.5):
    # Extract exposure (effective area) and energy bins
    myexp = dataset.exposure
    exposure = myexp.data[:, 0, 0]
    energies = myexp.geom.axes['energy_true'].center
    widths = myexp.geom.axes['energy_true'].edges_max - myexp.geom.axes['energy_true'].edges_min
    # Compute cumulative sum of the effective area weighted by bin widths
    cumulative_sum = np.cumsum(exposure * widths)
    # Normalize to a cumulative distribution (range 0 to 1)
    cumulative_sum /= cumulative_sum[-1]
    # Interpolate to find the energy corresponding to the 'factor' quantile
    emin = np.interp(factor, cumulative_sum, energies)
    emax = np.interp(1 - factor, cumulative_sum, energies)
    return (emin, emax)

'''
# Assuming flat SED
def get_weighted_mean_energy_from_aeff(dataset):
    # Extract exposure (effective area) and energy bins
    myexp = dataset.exposure
    exposure = myexp.data[:, 0, 0]
    energies = myexp.geom.axes['energy_true'].center
    widths = myexp.geom.axes['energy_true'].edges_max - myexp.geom.axes['energy_true'].edges_min
    # Calculate the total effective area (weighted by bin width)
    total_effective_area = np.sum(exposure * widths)
    # Calculate the weighted mean energy
    weighted_mean_energy = np.sum(energies * exposure * widths) / total_effective_area
    return weighted_mean_energy
'''

# Weighting by photon flux (assuming constant photon flux source)
def get_weighted_mean_energy_from_aeff(dataset):
    # Extract exposure (effective area) and energy bins
    myexp = dataset.exposure
    exposure = myexp.data[:, 0, 0]
    energies = myexp.geom.axes['energy_true'].center
    widths = myexp.geom.axes['energy_true'].edges_max - myexp.geom.axes['energy_true'].edges_min
    # Weight by photon flux (1/E)
    photon_flux_weight = 1 / energies
    # Total effective area weighted by photon flux
    total_effective_area = np.sum(exposure * widths * photon_flux_weight)
    # Weighted mean energy with photon flux weighting
    weighted_mean_energy = np.sum(energies * exposure * widths * photon_flux_weight) / total_effective_area
    return weighted_mean_energy

def calculate_assym_errors(cen,err,unit = 'erg/(cm*cm*s)'):
    units = u.Unit(unit)
    cen = cen.to(unit).value
    err = err.to(unit).value
    ypos = cen+err
    yneg = 10**(2*np.log10(cen)-np.log10(cen+err))
    return(cen*units,yneg*units,ypos*units)

def draw_sed_contours(model,energy_edges,color,label,facealpha=0.5,edgealpha=1,ax=None,x=None,y=None,yerr=None,**kwargs):
    if ax is None:
        ax = model.plot_error(energy_edges,sed_type='e2dnde',facecolor='white')
    
    # Block to draw the model with custom style
    if model==None:
        ene=x
        cen=y
        err=yerr
    else:
        ene = np.geomspace(energy_edges[0],energy_edges[-1],100)
        cen,err = model.evaluate_error(energy=ene)*ene*ene
    
    cen,yneg,ypos = calculate_assym_errors(cen,err)
    ax.fill_between(ene,yneg,ypos,
                    facecolor=(color,facealpha),
                    edgecolor=(color,edgealpha),
                    label=label,**kwargs)
    return(ax)

def contains_one_of(string,list_strings):
    for s in list_strings:
        if s in string:
            return(True)
    return(False)