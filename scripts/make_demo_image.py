from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

from src.astronomy import altaz_to_vector
from src.solver import project_vectors, SolveResult

ROOT=Path(__file__).resolve().parents[1]
STARS=pd.read_csv(ROOT/'data/stars_seed.csv')


def gmst_deg(dt):
    # Meeus-style approximate GMST for demo-data generation.
    jd=(dt.timestamp()/86400.0)+2440587.5
    T=(jd-2451545.0)/36525.0
    return (280.46061837+360.98564736629*(jd-2451545.0)+0.000387933*T*T-T*T*T/38710000.0)%360

def radec_to_altaz(ra_h,dec,dt,lat,lon):
    lst=(gmst_deg(dt)+lon)%360
    ha=np.deg2rad((lst-ra_h*15+540)%360-180)
    dec=np.deg2rad(dec); lat=np.deg2rad(lat)
    alt=np.arcsin(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha))
    az=np.arctan2(-np.sin(ha)*np.cos(dec), np.sin(dec)*np.cos(lat)-np.cos(dec)*np.sin(lat)*np.cos(ha))
    return np.rad2deg(az)%360, np.rad2deg(alt)


def render():
    dt=datetime(2026,1,15,2,0,tzinfo=timezone.utc); lat=35.0; lon=-80.0
    az=[]; alt=[]
    for _,r in STARS.iterrows():
        a,e=radec_to_altaz(r.ra_h,r.dec_deg,dt,lat,lon); az.append(a); alt.append(e)
    az=np.array(az); alt=np.array(alt)
    valid=alt>8
    vec=altaz_to_vector(az[valid],alt[valid])
    # Aim near Betelgeuse's calculated alt/az.
    idx=int(STARS.index[STARS.name=='Betelgeuse'][0]); az0,alt0=az[idx],alt[idx]
    # Create a landscape phone-like image.
    W,H=960,720
    fov=48.0; aspect=H/W
    params=(az0,alt0,8.0,fov)
    from src.solver import project_vectors
    proj,visible_proj=project_vectors(vec,*params,aspect=aspect)
    proj=proj[visible_proj]
    img=np.random.default_rng(2026).normal(2,1.8,(H,W)).astype(np.float32)
    valid2=np.where(valid)[0][visible_proj]
    for (xn,yn),ix in zip(proj,valid2):
        x=(xn+1)*0.5*(W-1); y=(1-yn)*0.5*(H-1)
        mag=float(STARS.iloc[ix].mag)
        strength=max(60,220*np.exp(-0.5*mag))
        sigma=max(1.0,min(3.2,1.5+0.1*(mag+1)))
        yy,xx=np.mgrid[:H,:W]
        img += strength*np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
    img=np.clip(img,0,255).astype(np.uint8)
    out=Image.fromarray(img,'L').convert('RGB')
    out.save(ROOT/'data'/'demo_orion.png')
    print('saved',ROOT/'data'/'demo_orion.png', 'center',az0,alt0)
if __name__=='__main__': render()
