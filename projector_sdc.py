# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
from email.mime import image
from email.policy import default
import os
from time import perf_counter

import click
import time
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import glob
from OSU import encoder, gestos
from matplotlib import pyplot as plt
from pathlib import Path


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', type=str, default='models/ffhq2.pkl',show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--images_path',            help='Path to images directory', type=str, default='img/', show_default=True)
@click.option('--outdir',                 help='Path to output directory', type=str, default='out/', show_default=True)

def run_projection(
    network_pkl: str,
    outdir: str,
    seed: int,
    images_path: str
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector_sdc.py --images_path=img/ --outdir=out/
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)

    # guardo una lista de las imagenes que ya tienen proyeccion para no hacerla dos veces
    imagenes_proyectadas = []

    # Me fijo si hay imagenes nuevas en la carpeta y las ordeno por fecha
    new = glob.glob(os.path.join(images_path, '*.png'))
    #remove duplicates from images
    images_list = list(dict.fromkeys(new))
    images_list.sort(key=lambda x: os.path.getmtime(x))

    while True:
        # Me fijo si hay imagenes nuevas en la carpeta y las agrego atrás
        new = glob.glob(os.path.join(images_path, '*.png'))
        #remove duplicates from images
        images_list = list(dict.fromkeys(new))
        images_list.sort(key=lambda x: os.path.getmtime(x))
        # Load target images.

        for img in range(0,len(images_list)):
            print(f'Hay {len(images_list)- len(imagenes_proyectadas)} imagenes por proyectar')
            
            if img not in imagenes_proyectadas:
                print('Proyectando imagen "%s"...' % images_list[img])
                #try:
                    #Load target image.
                G = images_list[img]

                # Optimize projection.
                start_time = perf_counter()

                projected_w, npz, weight_deltas = encoder(G)
                print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

                # get images_list[img] filename
                filename = Path(images_list[img]).stem
                os.makedirs(outdir, exist_ok=True)

                projected_w.save(f'{outdir}/{filename}.png')
                print('Generando gestos...')

                start_time = perf_counter()
                steps = 10
                n = 3
                images_react = gestos(npz,weight_deltas, steps, n)
                print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
                j = 1
                os.makedirs(f'{outdir}/{filename}', exist_ok=True)
                for gesture in images_react:
                    # make grid of image list
                    

                    for face in gesture:
                        face.save(f'{outdir}/{filename}/{j}.png')
                        j += 1
                print(f'Terminé con {filename}.')
                imagenes_proyectadas.append(img)
                # except Exception as e: 
                #     print(e)
                #     pass
            # Si ya se proyectaron todas las imagenes, espero 5 segundos y vuelvo a chequear
            time.sleep(5)

#----------------------------------------------------------------------------

if __name__ == "__main__":

    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
