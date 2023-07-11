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

from pqsar2cpd import pqsar2cpd, create_datasets
import fire
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def train(
    train_set: tf.data.Dataset,
    valid_set: tf.data.Dataset,
    epochs: int,
    compound_dim: int,
    profile_dim: int
):
    """Main training function.
    It trains the model by using profiles as condition given to the generator
    along the random Gaussian noise. To validate if the generator outputs
    fullfill the condition, a profile network predictions are added the
    discriminator loss. Discriminator itself only predicts if the molecules in
    the latent space originate from the real chemical embedding distribution,
    while the profile network predicts the same for combined chemical and
    profile space.

    Validation of the model while training is done each epoch. Fake output from
    the generator built on the validation set is fed to the profile network to
    make predictions that are averaged over the whole validation set.

    To prevent mode collapse during the cGAN training, several tricks are
    employed. Soft labels are used instead of binary labels, and the model is
    trained on real and fake batches separately. Also, the generator is trained
    only every 10 steps, while the discriminator and the profile network is
    updated during every step. Lastly, the generator is trained with fake
    samples but real labels.

    Args:
        train_set: tensorflow dataset for training
        valid_set: tensorflow dataset for validation
        epochs: number of epochs for the training
        compound_dim: length of compound vectors
        profile_dim: length of profile vectors
    """
    print(">>> Initializing models")
    model = pqsar2cpd(compound_dim, profile_dim)

    print(">>> Training")
    for epoch in range(epochs):
        for step, batch in enumerate(train_set):
            x_real, profiles, y_real, y_fake = batch

            latent_points = tf.random.normal(
                (x_real.shape[0], x_real.shape[1]), 0, 1, tf.float32
            )
            x_fake = model.generator.predict([profiles, latent_points])

            d_loss1, _ = model.discriminator.train_on_batch([x_real], y_real)
            d_loss2, _ = model.discriminator.train_on_batch([x_fake], y_fake)
            d_loss = 0.5 * np.add(d_loss1, d_loss2)

            profile_loss1, _ = model.profile_network.train_on_batch(
                [profiles, x_real], y_real
            )
            profile_loss2, _ = model.profile_network.train_on_batch(
                [profiles, x_fake], y_fake
            )
            profile_loss = 0.5 * np.add(profile_loss1, profile_loss2)

            if (step % 10 == 0) and (step != 0):
                g_loss = model.cgan.train_on_batch(
                    [x_fake, profiles], [y_real, y_real]
                )

        accuracy = 0
        for step, batch in enumerate(valid_set):
            x_real, profiles, y_real, y_fake = batch
            latent_points = tf.random.normal(
                (x_real.shape[0], x_real.shape[1]), 0, 1, tf.float32
            )
            x_fake = model.generator.predict([profiles, latent_points])
            accuracy += tf.math.reduce_mean(
                model.profile_network.predict([profiles, x_fake]).reshape(-1,)
            ).numpy()

        print(f"""{epoch}: train discriminator_loss: {d_loss}
                train generator_loss: {g_loss[0]} 
                validation profile accuracy: {accuracy/len(valid_set)}""")

    # only the generator gets saved for inference
    model.generator.save("pqsar2cpd.h5")
    print('>>> Done')


def main(
    compounds: str,
    profiles: str,
    epochs: int = 200
):
    """Main function invoked by the Fire package.
    All required arguments can be given when running the script:
    e.g. python train.py --compounds='cpds.npy' --profiles='profiles.npy
    
    Optional argument 'epochs' does not have to be given, unless you want a 
    different value than default by specifying e.g. --epochs=400

    Args:
        compounds: path to the compound numpy array
        profiles: path to the profile numpy array
        epochs: number of epochs for the training
    """
    print(">>> Loading files")
    compounds = np.load(compounds)
    profiles = np.load(profiles)
    assert compounds.shape[0] == profiles.shape[0]
    print(">>> Creating datasets")
    dataset = create_datasets(compounds, profiles)
    train_len = int(len(dataset)*0.9)
    train_set = dataset.take(train_len)
    valid_set = dataset.skip(train_len)
    train(train_set, valid_set, epochs, compounds.shape[1], profiles.shape[1])


if __name__ == "__main__":
    fire.Fire(main)
