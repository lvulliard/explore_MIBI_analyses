#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import tifffile, time
import collections
import numpy as np
from PIL import Image
import pandas as pd

import logging
clog = logging.getLogger("cellpose")
clog.propagate = False
from cellpose import models

print(time.ctime(), "- Starting script.", flush=True)

rootfolder = "./image_data/"
imgfolders = [x for x in os.listdir(rootfolder) if x[0] != "."]

# For the sake of simplicity we work with regions defined by individual nuclei (which is missing all cytoplasm info).
nucmodel = models.Cellpose(gpu=False, model_type='nuclei')

# These channels do not contain unique biological information
chans_to_exclude = ["Au.tiff", 'chan_39.tiff',
                    'chan_48.tiff', 'chan_69.tiff', 'chan_71.tiff',
                    'membrane.tiff', 'nuclei.tiff']

# For each image set (= sample)
for f in imgfolders:
    print(f"{time.ctime()} - Processing {f}.", flush=True)
    with Image.open(f"{rootfolder}{f}/rescaled/nuclei.tiff") as im:
        # Normalize for visualization
        nuc_data = im/np.max(im)
    
    segmentation = nucmodel.eval(
        nuc_data, diameter=15, channels=[1, 0], resample=False
    )
    tifffile.imwrite(f"cellpose_nucseg_{f}.tiff", (segmentation[0] > 0).astype("float32"))
        
    pixels_per_nucleus = np.unique(segmentation[0], return_counts=True)
    # How many nuclei kept for this example?
    nucleus_ids = pixels_per_nucleus[0][1:][pixels_per_nucleus[1][1:] > 90] # Discard speckles
    tifffile.imwrite(f"cellpose_nucseg_filtered_{f}.tiff", (np.isin(segmentation[0], nucleus_ids)).astype("float32"))

    print(f"{time.ctime()} - {len(nucleus_ids)} segmentation masks kept. Extracting intensities.", flush=True)
        
    chans = [x[:-5] for x in os.listdir(f"{rootfolder}{f}/rescaled/") if x not in chans_to_exclude]
    
    # We want a DF with rows = nuclei and columns = intensities
    markers = pd.DataFrame({c: [-1 for _ in nucleus_ids] for c in chans})
    markers.set_index(nucleus_ids, inplace = True)
    # Used to store coordinates
    markers["x"] = -1
    markers["y"] = -1

    for chan in chans:
        # Read a channel
        chan_data = np.array(Image.open(f"{rootfolder}{f}/rescaled/{chan}.tiff"))
        # Look at a nucleus
        for n in nucleus_ids:
            n_mask = segmentation[0] == n
            markers.loc[n, chan] = np.median(chan_data[n_mask])
            # Add nucleus coordinates
            markers.loc[n, "x"] = np.median(np.argwhere(n_mask)[:,0])
            markers.loc[n, "y"] = np.median(np.argwhere(n_mask)[:,1])
            
    # Export marker table
    markers.to_csv(f"marker_intensities_{f}.csv")