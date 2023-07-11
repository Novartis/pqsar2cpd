# pqsar2cpd - de novo generation of hit-like molecules from pQSAR pIC50 with AI-based generative chemistry

[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![tensorflow](https://img.shields.io/badge/TensorFlow-2.8-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org) [![LICENSE](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Novartis/pqsar2cpd/blob/main/LICENSE)

This repository contains the code of the conditional generative adversarial network capable of translating [pQSAR](https://github.com/Novartis/pQSAR) profiles of pIC50 values into novel chemical structures, as described in [[1]](https://www.biorxiv.org/content/10.1101/2021.12.10.472084v1)

The model itself operates entirely in the latent space. This means users can use any external molecular encoder/decoder to encode the molecules into vectors for training, and decode the output back to SMILES after inference. This way, pqsar2cpd can be implemented into any existing pipeline seamlessly. We have succesfully tested the approach with [CDDD](https://github.com/jrwnter/cddd), [JT-VAE](https://github.com/wengong-jin/icml18-jtnn), [HierVAE](https://github.com/wengong-jin/hgraph2graph), and [MoLeR](https://github.com/microsoft/molecule-generation). 

Since the model is input-agnostic, other property profiles, such as gene expression profiles or protein embeddings, could potentially be used instead of pQSAR to generate novel compounds.

## Requirements
pqsar2cpd is implemented in Tensorflow. To make sure all your packages are compatible, you can install the dependencies using the provided requirements file:
```
pip install -r requirements.txt
```

## Training
To train a new model, you need a set of compound vectors coming from a molecular encoder, and a matching set of property profiles. The compound and profile sets should be separate numpy arrays containing n-dimensional vectors, one row per compound, with 1:1 correspondence in indexing. If you're interested in using pQSAR profiles, you can follow the instructions in the [pQSAR](https://github.com/Novartis/pQSAR) repository.

To use the model out of the box, save the compounds and profiles as separate .npy files with NumPy.

To train the model, run:

```
python train.py --compounds='cpd.npy' --profiles='profiles.npy'
```
you can also specify an optional argument for the number of epochs, e.g. `--epochs=400`.

The script will train the cGAN, and save the generator as pqsar2cpd.h5, which will be ready for use in inference.

## Inference
To generate novel molecules out of a set of profiles, run:

```
python predict.py --model='pqsar2cpd.h5' --profiles='test.npy' --output='new_mols.h5' --n_samples=100
```
This will load the profile numpy array from `test.npy` and will generate 100 samples for each of the profiles in the set. Then, the results will be saved in `new_mols.h5` in hdf5 format, with the samples stored as a dataset with the profile index as key. These can now be passed to the molecular decoder to get the SMILES.

## Contact
Code authored by [Michal Pikusa](mailto:michal.pikusa@novartis.com)

Contributions: **Florian Nigsch**, **W. Armand Guiguemde**, Eric Martin, William J. Godinez, Christian Kolter

## References
```
[1] De-novo generation of novel phenotypically active molecules for Chagas disease from biological signatures using AI-driven generative chemistry
Michal Pikusa, Olivier René, Sarah Williams, Yen-Liang Chen, Eric Martin, William J. Godinez, Srinivasa P S Rao, W. Armand Guiguemde, Florian Nigsch
bioRxiv 2021.12.10.472084; doi: https://doi.org/10.1101/2021.12.10.472084
`````