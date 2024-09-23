import requests
from astropy.table import Table
import json
from astroquery.sdss import SDSS

class PS1_Catalog(object):
    BASE_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"
    RELEASES = {"dr1": ("mean", "stack"), "dr2": ("mean", "stack", "detection")}
    FORMATS = {"csv", "votable", "json"}

    def __init__(self,ra=None,dec=None,radius=None):
        if (ra,dec,radius) != (None,None,None):
            self.table = self.query_catalog(ra,dec,radius)

    def ps1cone(self, ra, dec, radius, table="mean", release="dr2",
                format="csv", columns=None, verbose=False, **kwargs):
        """Perform a cone search in the PS1 catalog."""
        params = {"ra": ra, "dec": dec, "radius": radius, **kwargs}
        return self.ps1search(
            table=table, release=release, format=format, columns=columns,
            verbose=verbose, **params
        )

    def ps1search(self, table="mean", release="dr2", format="csv",
                  columns=None, verbose=False, **kwargs):
        """Perform a general search in the PS1 catalog."""
        self._check_legal(table, release)
        if format not in self.FORMATS:
            raise ValueError(f"Invalid format: must be one of "
                             f"{', '.join(self.FORMATS)}")
        
        url = f"{self.BASE_URL}/{release}/{table}.{format}"
        params = kwargs.copy()
        
        if columns:
            available_columns = {col.lower() for col in
                                 self.ps1metadata(table, release)['name']}
            invalid_columns = [col for col in columns if
                               col.lower().strip() not in available_columns]
            if invalid_columns:
                raise ValueError(f"Invalid columns: {', '.join(invalid_columns)}")
            params['columns'] = ','.join(columns)
        
        response = requests.get(url, params=params)
        if verbose:
            print(response.url)
        response.raise_for_status()
        return response.json() if format == "json" else response.text

    def _check_legal(self, table, release):
        """Validate the table and release values."""
        if release not in self.RELEASES:
            raise ValueError(f"Invalid release: must be one of "
                             f"{', '.join(self.RELEASES)}")
        if table not in self.RELEASES[release]:
            raise ValueError(f"Invalid table for {release}: must be one of "
                             f"{', '.join(self.RELEASES[release])}")

    def ps1metadata(self, table="mean", release="dr2"):
        """Return metadata for the specified catalog and table."""
        self._check_legal(table, release)
        url = f"{self.BASE_URL}/{release}/{table}/metadata"
        response = requests.get(url)
        response.raise_for_status()
        metadata = response.json()
        return Table(rows=[(item['name'], item['type'], item['description'])
                           for item in metadata],
                     names=('name', 'type', 'description'))

    def mast_query(self, request):
        """Perform a MAST query."""
        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-agent": f"python-requests/{'.'.join(map(str, sys.version_info[:3]))}"
        }
        response = requests.post("https://mast.stsci.edu/api/v0/invoke",
                                 data={"request": json.dumps(request)},
                                 headers=headers)
        response.raise_for_status()
        return response.headers, response.text

    def resolve(self, name):
        """Get RA and Dec for an object using the MAST name resolver."""
        request = {
            'service': 'Mast.Name.Lookup',
            'params': {'input': name, 'format': 'json'}
        }
        headers, response_text = self.mast_query(request)
        resolved = json.loads(response_text)
        try:
            coord = resolved['resolvedCoordinate'][0]
            return coord['ra'], coord['decl']
        except (IndexError, KeyError):
            raise ValueError(f"Unknown object '{name}'")

    def query_catalog(self, ra, dec, radius):
        """Query the catalog and return results as an astropy Table."""
        constraints = {
            'nDetections.gt': 10,
        }
        for f in ['g','r','z','i']:
            add = {
                f'{f}MeanPSFMag.gt': 10,
                f'{f}MeanPSFMag.lt': 21,
                f'{f}MeanPSFMagErr.gt': 0.0,
                f'{f}MeanPSFMagErr.lt': 0.2,
                f'{f}MeanPSFMagStd.gt': 0.0,
                f'{f}MeanPSFMagStd.lt': 0.2,
                f'{f}QfPerfect.gt':0.85,
            }
            constraints = {**constraints, **add}
            
        columns = [
            "objID", "raMean", "decMean", "nDetections", "ng", "nr", "ni",
            "nz", "ny", "qualityFlag", "gMeanPSFMag", "rMeanPSFMag",
            "iMeanPSFMag", "zMeanPSFMag", "yMeanPSFMag", "gMeanPSFMagErr",
            "rMeanPSFMagErr", "iMeanPSFMagErr", "zMeanPSFMagErr",
            "yMeanPSFMagErr", "gMeanPSFMagStd", "rMeanPSFMagStd",
            "iMeanPSFMagStd", "zMeanPSFMagStd", "yMeanPSFMagStd"
        ]
        results = self.ps1cone(ra, dec, radius, release='dr2', columns=columns,
                               verbose=True, **constraints)
        return Table.read(results, format='ascii.csv')

class SDSS_Catalog(object):
    BASE_URL = "https://cas.sdss.org/dr18/SearchTools/sql"
    BASE_URL = "https://cas.sdss.org/dr18/en/tools/search/x_results.aspx"

    def __init__(self,ra=None,dec=None,radius=None):
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        if (ra,dec,radius) != (None,None,None):
            self.table = self.query_star_photometry(ra,dec,radius)

    def query_star_photometry(self, ra, dec, radius, max_rows=100000):
        """Query for clean star photometry around a given position."""
        sql_query = (
            "SELECT TOP {max_rows} "
            "G.objID, G.run, G.rerun, G.camcol, G.field, G.obj, G.type, G.ra, G.dec, "
            "G.u, G.g, G.r, G.i, G.z, G.Err_u, G.Err_g, G.Err_r, G.Err_i, G.Err_z, G.flags_r "
            "FROM Star AS G "
            "JOIN dbo.fGetNearbyObjEq({ra}, {dec}, {radius}) AS GN "
            "ON G.objID = GN.objID "
            "WHERE ((G.flags_r & 0x10000000) != 0) "  # CLEAN flag, detected in BINNED1
            "AND ((G.flags_r & 0x8100000c00a4) = 0) "  # not EDGE, NOPROFILE, not SATURATED, etc.
            "AND (((G.flags_r & 0x400000000000) = 0) OR (G.psfmagerr_r <= 0.2)) "  # not DEBLEND_NOPEAK or small PSF error
            "AND (((G.flags_r & 0x100000000000) = 0) OR (G.flags_r & 0x1000) = 0) " # not INTERP_CENTER or COSMIC_RAY
            "AND ((G.u < 21) AND (G.g < 21) AND (G.r < 21) AND (G.z < 21) AND (G.i < 21))"
            # Not too dim, not too much error in the photometry
        ).format(max_rows=max_rows, ra=ra, dec=dec, radius=radius)
        return SDSS.query_sql(sql_query)