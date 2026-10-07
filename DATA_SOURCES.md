# Data sources and acknowledgements

## HYG Stellar Database v4.1

The first-release catalog loader uses HYG v4.1 as the primary visible-star catalog and caches a trimmed local subset. The HYG project README documents the fields (including RA, Dec, apparent magnitude, spectral type, constellation, proper name) and states that the current v4.1 work is licensed under Creative Commons Attribution-ShareAlike 4.0.

Project page: https://github.com/astronexus/HYG-Database

## ESA Gaia

Gaia is the recommended source for a later, higher-precision catalog extension. The current application does not download Gaia's full catalog because that is unnecessary for a smartphone semester project.

Archive: https://gea.esac.esa.int/archive/

## CDS SIMBAD / VizieR

SIMBAD and VizieR are useful for cross-identifications and published catalogs when extending the list of named targets or adding deep-sky objects.

SIMBAD: https://simbad.cds.unistra.fr/simbad/

## Planet positions

The current implementation uses Astropy's coordinate and solar-system tooling locally. A later research version can switch the planet layer to NASA/JPL Horizons for explicit ephemeris queries.

JPL Horizons: https://ssd.jpl.nasa.gov/horizons/
