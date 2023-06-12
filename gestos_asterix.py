# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import glob
from PIL import Image
import sys
from pathlib import Path

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

# X in [N, 18, 512], N elements per class
def center_of_mass(X):
    X = X.reshape(X.shape[0], -1)
    return np.mean(X, axis=0).reshape(18,512)


# given w neutral latent
def generate_images(w):

    # Levantar proyecciones de OSU por clase
    ...
    # Generate center of mass
    centers = []
    for e in emotions:
        centers.append(center_of_mass(X[e]))


    for i in range(0,apex,step):
        d = dirr_todas[i]   
        d = torch.tensor(d, device=device)  
        ws = ws = proj_todas[l].reshape(1,18,512)
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws): 
                # por cada direccion movemos la imagen 2/5 en esa direccion  
                W = w+((d)*(2/5))  
                #w = w + ((d-w)*(i/5))                               
                img = G.synthesis((W).unsqueeze(0), noise_mode=noise_mode)                   
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}'+'_'+str(i)+'.png') 
                #np.savez(f'{outdir}/projected_w{i}.npz', w=W.cpu().numpy())               
                print("foto " + str(i)+' esta lista')
                
                    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------




# python gestos_sdc.py --gesto=otro
#python gestos_sdc.py  









