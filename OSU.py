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
import warnings
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

def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0][len('LR_model'):]
    return int(int_part) 

print('starting')

def run_alignment(image_path):
    import dlib
    from scripts.align_faces_parallel import align_face
    predictor = dlib.shape_predictor("./predictor/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image
experiment_type = 'faces'

warnings.filterwarnings('ignore')

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
    
def gestos(latent_vector, weights_deltas, steps, n):
    emotions = []
    for e in range(1,n):
        modelo_emocion_i = sorted(glob.glob(f'./models/m2/REG{e}/*.sav'), key=get_key)
        modelos = [x for x in modelo_emocion_i]
        latent_vector = latent_vector.reshape([18,512])
        inicial = torch.tensor(latent_vector, device=torch.device('cuda')).reshape([18,512])
        modelo_e = inicial[0].reshape(1,-1)
        f = []
        for j in range(len(modelos)):
            loaded_model = pickle.load(open(modelos[j], 'rb'))
            f.append(loaded_model.predict(modelo_e[:,j].reshape(1,-1).detach().cpu().numpy()))
        
        result = np.asarray(f+f+f+f+f+f+f+f+f+f+f+f+f+f+f+f+f+f)
        final = torch.tensor(result, device=torch.device('cuda')).reshape([18,512])
        emotions.append(final)

    delta = 9
    emotion_array = []
    for j in range(n):
        image_list = []
        for i in range(steps): 
            final = emotions[j-1]
            W = inicial + ((final-inicial)*(2*i/delta))
            lat = [W.reshape([1,18,512]).float()]     
            lat_image = latent_editor._latents_to_image(all_latents= lat, weights_deltas=weights_deltas)           
            res = (lat_image[0])[0]
            image_list.append(res)
        emotion_array.append(image_list)
    return emotion_array
    
def gestos_direcciones(latent_vector, weights_deltas, steps, n):
    source_dir = Path('./models/directions/')
    dirr = source_dir.glob('*.npz')
    dirr_todas = [torch.tensor(np.load(x)['w']).reshape([18,512]).cuda() for x in sorted(dirr)]
    directions = dirr_todas
    latent_vector = latent_vector.reshape([18,512])
    inicial = torch.tensor(latent_vector, device=torch.device('cuda')).reshape([18,512])
    delta = 10
    emotion_array = []
    for i in range(n):
        print(f'Generando gesto {i+1} de {n}')
        d = directions[i]
        d = torch.tensor(d,device='cuda')
        image_list = []
        for i in range(steps):
            W = inicial + (d * i/delta)
            lat = [W.reshape([1,18,512]).float()]     
            lat_image = latent_editor._latents_to_image(all_latents= lat, weights_deltas=weights_deltas)           
            res = (lat_image[0])[0]
            image_list.append(res)
        emotion_array.append(image_list)
    return emotion_array

def create_grid(imgs):
        widths, heights = zip(*(i.size for i in imgs))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        return new_im
        
        
