import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import glob
import pickle
import torch

def chunks(xs, n):
    n = max(1, n)
    
    return [xs[i:i+n] for i in range(0, len(xs), n)]
    
#paths a carpetas
gestos = Path('./Directions/all')

# glob
gestos_npz = gestos.glob('*.npz')
npz = [(torch.tensor(np.load(x)['w']).reshape([9216,1]).cuda()).detach().cpu().numpy() for x in sorted(gestos_npz)] 
emotion = chunks(npz,230)
print(type(emotion))

X = np.empty(shape=(9216, 1))
for z in emotion[0]:
    X = np.concatenate((X, z), axis=0)
    print(X)

#X = array de caras neutras(Alargadas), y es array de caras felices alargadas
for k in range(len(emotion)):

	y = np.empty(shape=(9216, 1))
	for z in emotion[k]:
	    y = np.concatenate((y, z), axis=0)
	    
	reg = LinearRegression().fit(X, y)
	print('score = ',reg.score(X, y))

	# aca dado un array alargado de una cara neutra nos predice el array alargado de cara con emocion correspondiente
	# save the model to disk
	filename = f'./models/LR/LR_model{k+1}.sav'
	pickle.dump(reg, open(filename, 'wb'))
	print('modelo ',k,' terminado')

