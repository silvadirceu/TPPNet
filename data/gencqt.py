# Generate CQT from wavefile
# Save as npy

import librosa
import numpy as np
import os
import glob
from multiprocessing import Pool, set_start_method

from tqdm import tqdm


def CQT(args):

    if not isinstance(args,list):
        args = [args]

    for k, in_file, out_file in args:

        print(f"{k} de {len(args)} -- {in_file}")

        try:
            data, sr = librosa.load(in_file, sr=22050.0, mono=True)
            if len(data)<1000:
                return
            cqt = np.abs(librosa.cqt(y=data, sr=sr))
            mean_size = 20
            height, length = cqt.shape
            new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
            for i in range(int(length/mean_size)):
                new_cqt[:,i] = cqt[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
            np.save(out_file, new_cqt)
        except :
            print('wa', in_file)


if __name__ == "__main__":

    #in_dir = '/Users/dirceusilva/Documentos/tests/setlist_ecad/audios'
    in_dir = '/mnt/dev/dirceusilva/dados/Cover/setlist_all/setlist_ecad/audio/universe'
    #out_dir = '/Users/dirceusilva/Documentos/tests/setlist_ecad/features/cqt'
    out_dir = '/mnt/dev/dirceusilva/dados/Cover/setlist_all/setlist_ecad/features/universe_cqt'
    parallel = False
    
    files = glob.glob(os.path.join(in_dir, "**/*.ogg"), recursive=True)
    
    set_start_method("fork")

    params =[]
    for ii, file in enumerate(files):  
        track_id = file.split('/')[-1].split('.')[0]
        work_id = file.split('/')[-2]
        out_path = os.path.join(out_dir, work_id)
        os.makedirs(out_path, exist_ok=True)
        
        out_file = os.path.join(out_path, track_id + '.npy')
        if not os.path.exists(out_file):
            params.append((ii, file, out_file))

    if parallel:
        pool = Pool(processes=None)
        pool.map(CQT, params)
        pool.close()
        pool.join()
    else:
        CQT(params)


