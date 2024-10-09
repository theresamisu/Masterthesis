# Masterthesis: Multi-modal Vision Transformers for Crop Mapping from Satellite Image Time Series

This repository contains the code for my master thesis. Three of the presented architectures were published at IGARSS 2024. See our paper on [arxiv](https://arxiv.org/pdf/2406.16513). 

<details><summary>Abstract</summary>
Using images acquired by different satellite sensors has been shown to improve classification
performance in the framework of crop mapping from satellite image time series (SITS). Current
state-of-the-art architectures utilize self-attention to process the temporal dimension and convo-
lutions for the spatial dimensions of SITS. Motivated by the success of purely attention-based
architectures in crop mapping from single-modal SITS, we introduce several multi-modal, multi-
temporal architectures based on the single-modal Temporo-Spatial Vision Transformer (TSViT).
In order to enable it to incorporate features from multiple modalities to produce a single predic-
tion, our architectures use either a modified token embedding or a modified temporal encoder. To
assess their effectiveness, we compare them with each other as well as to single-modal baselines
and two existing architectures from the literature. Experiments are conducted on the EOekoLand
dataset for multi-modal multi-temporal crop mapping. The data contains two optical modalities
of different spatial, spectral and temporal resolutions as well as a third modality consisting of
synthetic aperture radar (SAR) images. Results show that our proposed architectures achieve
clear improvements over single-modal baselines. Moreover, we find that directly deriving to-
kens from the fused input achieves better results than using a separate token embedding for each
modality and fuse later in the TSViT architecture. Among proposed architectures with separate
token embeddings for each modality, those with modality-specific temporal encoders outper-
form architectures that use a single temporal encoder. In a detailed ablation study we investigate
the effect of hyperparameters on the performance of our most successful architectures. Finally, it
is shown that the proposed architectures outperform two other crop mapping architectures from
the literature which further affirms their effectiveness for multi-modal crop mapping.
</details>


## Requirements
create and activate virtual environment with micromamba with python version "Python 3.7.12"
```
micromamba create -n "env_name" -f requirements.yml
micromamba activate "env_name"
```

## Dataset
The non-public EOekoLand dataset is used for the evaluation. 

## Training
training and testing is done in the main.py file. Parameters that can be configured are specified in the `config/*.yaml` files and can be changed there.
```
python main.py --config /path/to/configfile.yaml --gpu 2
```

The following parameters can also be passed via the command line which will override their values in the provided config file. They are here given with their standard values:
```
--backbone TSViT
--batchsize 8
--factor 1
```

Each training run is logged with a Tensorboard logger to a folder in the `log_eoeko` subdirectory. They are sorted by modalities and architecture and numbered consecutively. When a training is finished and the best model is evaluated on the test set, the metric results are stored in a csv file in the same folder. This contains the metric results on the validation set, the test set, the number of parameters and the average batch inference time for the test set. 


An existing training can be resumed by giving the checkpoint and the config file belonging to the checkpoint. 
```
python main.py --config /path/to/checkpoint/configfile.yaml --checkpoint /path/to/checkpoint/checkpointfile.ckpt  --gpu 2
```

### Evaluating
It can further be evaluated on the test set:
```
python evaluate.py --config /path/to/checkpoint/configfile.yaml --checkpoint /path/to/checkpoint/checkpointfile.ckpt  --gpu 2
```

### Predictions
A model checkpoint can also be used to classify all patches within the Bavaria 2020 tile in order to create a large classification map. This can be done with
```
python predict.py --config /path/to/checkpoint/configfile.yaml --checkpoint /path/to/checkpoint/checkpointfile.ckpt  --gpu 2
```

