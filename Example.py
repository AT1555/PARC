# Example script for running PARC on all objects in the MPCORB database for a single date and time of flight

# Instead of a full installation, PARC works just be havig the parc directory in the same working directory as this (or any) script

import numpy as np
import parc
import urllib
from astropy import table
from os.path import exists

def readtarget(target): #convert the epoch to a np.datetime64 object and return orbital parameters for later functions
    rawepoch=target['Epoch']
    epoch=f"{ord(rawepoch[0])-55}{rawepoch[1:3]}-"
    if rawepoch[3].isnumeric(): epoch+=f"0{rawepoch[3]}-"
    else: epoch+=f"{ord(rawepoch[3])-55}-"
    if rawepoch[4].isnumeric(): epoch+=f"0{rawepoch[4]}"
    else: epoch+=f"{ord(rawepoch[4])-55}"
    epoch=np.datetime64(epoch)
    return epoch,target['M'],target['Node'],target['Peri.'],target['Incl.'],target['e'],target['a']

def main():
    #download the MPCORB database if it is not already in the current directory
    if not exists('MPCORB.DAT.gz'): urllib.request.urlretrieve('http://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz','MPCORB.DAT.gz')
    print('Database Downloaded\nImporting Data...')
    #read the database into an Astropy Table object
    data=table.Table.read('MPCORB.DAT.gz',format='ascii.fixed_width',header_start=30,data_start=32,col_starts=(0,8,14,20,26,37,48,59,70,80,92,105,107,117,123,127,137,142,150,161,166,194),col_ends=(7,13,19,25,35,46,57,68,79,91,103,106,116,112,126,136,141,149,160,165,193,202))
    
    data=data[:10] #just do the first 10 objects as an example
    
    tof=180 #sample time of flight of 180 days
    date=np.datetime64('2050-01-01') #assume a sample launch date of Jan. 1, 2050
    planet='earth' #choose earth as a starting planet, 'mars' is the other available option
    
    twoburn_results=[parc.DeltaV2Burn(*readtarget(target),date,tof,planet=planet) for target in data] #compute two burn delta-v for each target in the database at the specified launch date and time of flight
    threeburn_results=[parc.DeltaV3Burn(*readtarget(target),date,tof,planet=planet) for target in data] #compute three burn delta-v for each target in the database at the specified launch date and time of flight
    
    print(twoburn_results) #print the computed two burn delta-v's in km/s for the first ten objects in the database
    print(threeburn_results) #print the computed three burn delta-v's in km/s for the first ten objects in the database

main()