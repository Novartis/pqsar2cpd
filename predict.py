"""
Copyright (c) 2023 Novartis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import fire
import h5py

def predict(
    model: str,
    profiles: str,
    output: str,
    n_samples: int,
    cpd_dim: int
):
    """Main prediction function.
    It loads the trained model and profile data, and generates new samples for 
    each of the profile by feeding the profile along with randomly sampled 
    Gaussian noise. Results are saved in hdf5.

    Args:
        model: path to the trained model
        profiles: path to the .npy file with profiles for prediction
        output_file: path to .h5 file to save the resulting molecules in
        n_samples: number of samples to generate per each profile
        cpd_dim: length of the compound vector
    """
    generator = tf.keras.models.load_model(model)
    profiles = np.load(profiles)

    print('>>> Generating molecules')
    f = h5py.File(output, 'w')
    for i, item in enumerate(profiles):
        print(f'Profile #{i}')
        latent_points = tf.random.normal(
                (n_samples, cpd_dim), 0, 1, tf.float32
            )
        X = generator.predict([np.array([item]*n_samples), latent_points])
        f.create_dataset('profile_'+str(i), data=np.unique(X, axis=0))
    f.close()

    print('>>> Done')

if __name__ == '__main__':
    fire.Fire(predict)