#Import Packages 
import time
import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import glob
import re
sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
#from utils.inference_utils import run_inversion
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model, load_generator
from editing.face_editor import FaceEditor

def run_inversion(inputs, net, opts, return_intermediate_results=False):
    y_hat, latent, weights_deltas, codes = None, None, None, None

    if return_intermediate_results:
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        results_deltas = {idx: [] for idx in range(inputs.shape[0])}
    else:
        results_batch, results_latent, results_deltas = None, None, None
        
    
    for iter in range(opts.n_iters_per_batch):
        y_hat, latent, weights_deltas, codes, _ = net.forward(inputs,
                                                              y_hat=y_hat,
                                                              codes=codes,
                                                              weights_deltas=weights_deltas,
                                                              return_latents=True,
                                                              resize=opts.resize_outputs,
                                                              randomize_noise=False,
                                                              return_weight_deltas_and_codes=True)
                                                              
                   
        # busca los gestos

        if "cars" in opts.dataset_type:
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        if return_intermediate_results:
            store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas)

        # resize input to 256 before feeding into next iteration
        if "cars" in opts.dataset_type:
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)
    
    if return_intermediate_results:
        return results_batch, results_latent, results_deltas

    return y_hat, latent, weights_deltas, codes

def store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas):
    print('encontradoooo',type(y_hat),y_hat.size())
    for idx in range(y_hat.shape[0]):
        results_batch[idx].append(y_hat[idx])
        results_latent[idx].append(latent[idx].cpu().numpy())
        results_deltas[idx].append([w[idx].cpu().numpy() if w is not None else None for w in weights_deltas])
        
print('starting')

def run_alignment(image_path):
    import dlib
    from scripts.align_faces_parallel import align_face
    predictor = dlib.shape_predictor("./predictor/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image
experiment_type = 'faces'

EXPERIMENT_DATA_ARGS = {
    "faces": {
        "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
        "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
        "image_path": "./notebooks/images/brian.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "cars": {
        "model_path": "./pretrained_models/hyperstyle_cars.pt",
        "w_encoder_path": "./pretrained_models/cars_w_encoder.pt",
        "image_path": "./notebooks/images/car_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "afhq_wild": {
        "model_path": "./pretrained_models/hyperstyle_afhq_wild.pt",
        "w_encoder_path": "./pretrained_models/afhq_wild_w_encoder.pt",
        "image_path": "./notebooks/images/afhq_wild_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

#Load LR Model
e = 1 #expresion numero 1 es happy
LR_model_path = (f'./models/LR{e}/')
sav = sorted(glob.glob(f'./models/LR{e}/*.sav'), key=lambda x:float(re.findall("(\d+)",x)[0]))
loaded_models=[]
for i in sav:
    loaded_models.append(pickle.load(open(i, 'rb')))
print('LR successfully loaded!')

#Load HyperStyle Model
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
print('Hypersytle successfully loaded!')
latent_editor = FaceEditor(net.decoder)
#pprint.pprint(vars(opts))

#devuelve la imagen proyectada y el npz
def encoder(image_path):
    
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((256, 256))

    #Align Image 
    input_is_aligned = False #@param {type:"boolean"}
    if experiment_type == "faces" and not input_is_aligned:
        input_image = run_alignment(image_path)
    else:
        input_image = original_image

    input_image.resize((256, 256))
    n_iters_per_batch = 5 #@param {type:"integer"}
    opts.n_iters_per_batch = n_iters_per_batch
    opts.resize_outputs = False 

    #Run Inference
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image) 

    with torch.no_grad():
        tic = time.time()
        #result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cpu().numpy(),
        y_hat1, latent1, weights_deltas1, codes1= run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                                net, 
                                                                opts,
                                                                return_intermediate_results=False)
        toc = time.time()
        
    # encoder part
    lat_image = latent_editor._latents_to_image(all_latents= latent1, weights_deltas = weights_deltas1)           
    res = (lat_image[0])[0]
    torch.cuda.empty_cache()
    return res,latent1.reshape([1,18,512]).cpu(), weights_deltas1
    
def gestos(latent_vector, weights_deltas):
    cant = 10
    delta = 9

    inicial = torch.tensor(latent_vector, device=torch.device('cuda')).reshape([18,512])      
    first_layer = inicial[0]
    f = []
    for j in range(512):
        f.append(loaded_models[j].predict(first_layer[j].reshape(-1,1).detach().cpu().numpy()))
    print('Prediccion de gestos exitosa!')
    final = np.asarray(f+f+f+f+f+f+f+f+f+f+f+f+f+f+f+f+f+f)
    final = torch.tensor(final, device=torch.device('cuda')).reshape([18,512])

    image_list = []
    for i in range(cant): 
        W = inicial + ((final-inicial)*(i/delta))
        lat = [W.reshape([1,18,512]).float()]     
        lat_image = latent_editor._latents_to_image(all_latents= lat, weights_deltas=weights_deltas)           
        res = (lat_image[0])[0]
        image_list.append(res)
    return image_list
    
      
        
        
