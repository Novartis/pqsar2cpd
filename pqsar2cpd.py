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
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Dropout, Concatenate, Multiply, Subtract
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np


class pqsar2cpd:
    """Main class for the pqsar2cpd model

    The model is a conditional generative adversarial network (cGAN) that
    generates novel compounds in latent space given a profile of activities.
    The model consists of the generator, discriminator, and the profile
    network. The generator is fed a Gaussian noise vector in the compound
    embedding space along with a property profile vector that is used for
    conditioning. It outputs a compound vector.

    The discriminator takes a compound vector from the generator, and
    predicts the probability of it coming from the real distribution using
    binary classification.

    The profile network takes both a compound vector from the generator, and
    the real profile used for generation. It outputs the probability of the
    generated compound vector coming from the similar distribution to the
    combined profile and compound space.

    pqsar2cpd has been heavily tested on conditional generation from pQSAR
    property profiles (https://github.com/Novartis/pQSAR) which contain pIC50
    values of compound activity. However, the architecture is input/output
    agnostic, so in theory it could be adapted to generate compounds from
    different data modalities, as long as they can be represented in the form
    of a latent vector.

    Attributes:
        compound_dim = length of compound vector
        profile_dim = length of profile vector
        generator = generator model
        discriminator = discriminator model
        profile_network = profile network model
        cgan = conditional GAN that brings together the other networks
    """
    def __init__(self, compound_dim: int, profile_dim: int):
        """Initializes the class by bulding the cGAN model.

        Args:
            compound_dim = length of compound vectors
            profile_dim = length of profile vectors
        """
        self.compound_dim = compound_dim
        self.profile_dim = profile_dim
        self.generator = self.define_generator()
        self.discriminator = self.define_discriminator()
        self.profile_network = self.define_profile_network()
        self.cgan = self.define_cgan()

    @staticmethod
    def _generator_latent_processing(dim: int) -> keras.Model:
        """Function to build a model for processing of compound and
        profile embeddings separately for the generator and downsize
        both to the same dimensionality to be concatenated downstream.

        Args:
            dim (int) - dimensions of embedding
        Returns:
            keras model
        """
        inputs = Input(shape = (dim,))
        x = Dense(512, input_dim = dim)(inputs)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = BatchNormalization()(x)
        x = Dense(128)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = BatchNormalization()(x)
        model = Model(inputs, x, name = "generator_latent_processing")
        return model

    @staticmethod
    def _profile_latent_processing(dim: int) -> keras.Model:
        """Function to build a model for processing of compound and
        profile embeddings separately for the profile network and downsize
        both to the same dimensionality to be concatenated downstream.

        Args:
            dim (int) - dimensions of embedding
        Returns:
            keras model
        """
        inputs = Input(shape=(dim,))
        x = Dense(512, input_dim = dim)(inputs)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dense(128)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dropout(0.4)(x)
        model = Model(inputs, x, name = "profile_latent_processing")
        return model

    def define_discriminator(self) -> keras.Model:
        """Function to build a discriminator model that processes compound
        embeddings only to predict the probability of the compound being
        sampled from the real distribution
        """
        inputs = Input(shape = (self.compound_dim,))
        x = Dense(256)(inputs)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dropout(0.4)(x)
        x = Dense(256)(inputs)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation = "sigmoid")(x)
        model = Model(inputs = inputs, outputs = x)
        model.compile(
            loss="binary_crossentropy",
            optimizer = SGD(learning_rate = 0.005),
            metrics=["accuracy"]
        )
        return model

    def define_generator(self) -> keras.Model:
        """Function to build a generator that takes both the compound and
        profile embeddings, process them separately and then concatenate
        the latent vectors for final generation.

        Tanh activation layer should be removed if the compound embeddings are
        not bound to -1,1, depending on the chosen encoder.
        """
        profile_model = self._generator_latent_processing(self.profile_dim)
        compound_model = self._generator_latent_processing(self.compound_dim)

        combinedInput = Concatenate()(
            [profile_model.output, compound_model.output]
        )
        x = Dense(256)(combinedInput)
        x = LeakyReLU(alpha = 0.1)(x)

        x = Dense(self.compound_dim, activation="tanh")(x)

        model = Model(
            inputs=[profile_model.input, compound_model.input], outputs = x
        )
        return model

    def define_profile_network(self) -> keras.Model:
        """Function to build a profile network model that processes compound
        and profile embeddings to predict the probability of the compound
        fulfilling the condition of the profile.

        SubMult+NN comparison function (Wang and Jiang, 2019) is used to
        concatenate inputs after separate processing.
        """
        profile_model = self._profile_latent_processing(self.profile_dim)
        compound_model = self._profile_latent_processing(self.compound_dim)

        sub = Subtract()([compound_model.output, profile_model.output])
        multi = Multiply()([compound_model.output, profile_model.output])
        both = Multiply()([sub, sub])
        combinedInput = Concatenate()([both, multi])

        x = Dense(32)(combinedInput)
        x = LeakyReLU(alpha = 0.1)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(
            inputs=[profile_model.input, compound_model.input], outputs= x
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(learning_rate = 0.005),
            metrics=["accuracy"]
        )
        return model

    def define_cgan(self) -> keras.Model:
        """Function to build the cGAN model using generator, discriminator,
        and the profile network. Both the discriminator and the profile network
        are trained by direct update calls from the main script, so their
        weights are set to not trainable.

        The model takes the profile and Gaussian noise as input, and outputs
        the classification from the discriminator and the profile network.
        """
        self.discriminator.trainable = False
        self.profile_network.trainable = False

        gen_profile, gen_noise = self.generator.input
        gen_output = self.generator.output

        gan_output = self.discriminator([gen_output])
        profile_output = self.profile_network([gen_profile, gen_output])

        model = Model([gen_noise, gen_profile], [gan_output, profile_output])

        model.compile(
            loss=["binary_crossentropy", "binary_crossentropy"],
            optimizer=Adam(learning_rate=0.005, amsgrad=True),
        )
        return model


def create_datasets(
    compounds: np.array,
    profiles: np.array,
    batch_size: int = 128,
    shuffle: bool = True
) -> tf.data.Dataset:
    """Function to create Tensorflow dataset containing compounds, profiles,
    and both real and fake labels. Compound and profile order needs to match.

    Arg:
        compounds: numpy array with one compound vector per molecule
        profiles: numpy array with one property vector per molecule
        batch_size: size of batches that data will be split into
        shuffle: bool to randomly shuffle the dataset for training

    Returns:
        tf.data.dataset containing all training data in a single object
    """
    buffer_size = profiles.shape[0]

    real_labels = tf.random.uniform(
        minval=0.7, maxval=1.2, shape=[profiles.shape[0]]
    )
    fake_labels = tf.random.uniform(
        minval=0.01, maxval=0.3, shape=[profiles.shape[0]]
    )

    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(compounds),
            tf.data.Dataset.from_tensor_slices(profiles),
            tf.data.Dataset.from_tensor_slices(real_labels),
            tf.data.Dataset.from_tensor_slices(fake_labels),
        )
    )

    if shuffle:
        return dataset.shuffle(buffer_size).batch(batch_size)
    else:
        return dataset