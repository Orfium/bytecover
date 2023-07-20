# ByteCover Implementation

This repository contains the code implementation of the architecture described in the [ByteCover Paper](https://arxiv.org/pdf/2010.14022v2.pdf)

## Description
This implementation is as close as possible to the ByteCover architecture descibed in the paper, but with certain assumptions. Below we summarize some ambiguities that we encountered in the authors' model description and some assumptions that we made in order to resolve them.

### Ambiguities and Assumptions
| Ambiguity | Assumption |
| :-------: | :--------: |
| There is no description for padding and trimming audio in order to be batched, but a reference to [paper [10]](https://arxiv.org/pdf/1911.00334.pdf) in section 3.1 | We used the Algorithm 1 described in the referenced [paper [10]](https://arxiv.org/pdf/1911.00334.pdf) that uses three dataloaders that pad and trim the audio to 100, 150 and 200 seconds respectively and apply time stretching |
| According to the section 2.1 description W dimension should be **W=T/16** instead of W=T/8 | We used **W=T/16** |
| There is no description for the triplet mining process | We used random sampling |
| There is no description on the exact number of classes used in the classification loss | We used 8858, as the number of the unique cliques in the SHS100K dataset
| The number of epochs is not specified | We trained for 100 epochs |

## Dataset
Below there is a description of the files that the [./data](./data/) folder contains:
* [shs100k.csv](./data/interim/shs100k.csv) Contains the information of the [list](https://github.com/NovaFrost/SHS100K/blob/master/list) file from [SHS100K](https://github.com/NovaFrost/SHS100K) dataset, along with a column `id`, that is a unique identifier of every file
* [versions.csv](./data/interim/versions.csv) contains in each row the ids, in a list, that belong to the same clique, along with their respective clique.
* [train_ids.npy](./data/splits/train_ids.npy), [val_ids.npy](./data/splits/val_ids.npy), [test_ids.npy](./data/splits/test_ids.npy) contain the train, val and test splits respectively. Because of unavailability issues we used 85132 songs from the dataset.

## Usage
The project requires poetry that can be installed using
```
pip install poetry
```
Then in order to install the project
```
poetry install
```
Before training make sure to check the [config.yaml](./config/config.yaml) file for the available options.

The *debug* option generates dummy data for debugging purposes.

If you want to train the model using real audio download the dataset using the `Video ID` column from the [shs100k.csv](./data/interim/shs100k.csv) file and name them using the `id` column. Then specify in the `dataset_path` of the [config.yaml](./config/config.yaml) file the location of the data that were downloaded and the format of the files in the `file_extension` option.

Once everything is configured the training can start using the following command
```
poetry run bytecover
```

## Model Checkpoint
We have made our best model checkpoint publicly available. You can access it through the following link:
https://orfium-research-papers.s3.amazonaws.com/bytecover/orfium-bytecover.pt
