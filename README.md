## Submodular optimization-based diverse paraphrasing and its effectiveness in data augmentation

Source code for [NAACL 2019](https://naacl2019.org/) paper: [Submodular optimization-based diverse paraphrasing and its effectiveness in data augmentation](https://www.aclweb.org/anthology/N19-1363)

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/DiPS/blob/master/images/dips_model.png" alt="Image" height="420" >
</p>

- Overview of DiPS during decoding to generate k paraphrases. At each time step, a set of N sequences V<sup>(t)</sup> is used to determine k &lt; N sequences (X<sup>∗</sup>) via submodular maximization . The above figure illustrates the motivation behind each submodular component. Please see Section 4 in the paper for details.

### Also on GEM/NL-Augmenter 🦎 → 🐍

- Please use/check `diverse_paraphrase` in NL-Augmenter for the transformer-model version. [Diverse-Paraphrase: NL-Augmenter](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/diverse_paraphrase).

### Dependencies

- compatible with python 3.6
- dependencies can be installed using `requirements.txt`

### Dataset

Download the following datasets:

- [Quora](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ashutosh_iisc_ac_in/EeQ9jevrqJNNnFNsjKQR9VYBlePoAuZN2CSXobyXzCA0ew?e=9Vn0yw)
- [Twitter](https://indianinstituteofscience-my.sharepoint.com/:u:/g/personal/ashutosh_iisc_ac_in/EZe7CE3Ip0NOvBYAEjYR5RcBMyG-SjKeMI-XC6-njZrLGQ?e=gyDdGf)

Extract and place them in the `data` directory. Path : `data/<dataset-folder-name>`. 
A sample dataset folder might look like `data/quora/<train/test/val>/<src.txt/tgt.txt>`.

- Download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) into the `data` directory. In case the above link doesn't work, find the zip file [here](https://code.google.com/archive/p/word2vec/)

## Setup:

To get the project's source code, clone the github repository:

```shell
$ git clone https://github.com/malllabiisc/DiPS
```

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages:

```shell
$ pip install -r requirements.txt
```

Install the submodopt package by running the following command from the root directory of the repository:

```shell
$ cd ./packages/submodopt
$ python setup.py install
$ cd ../../
```

### Training the sequence to sequence model

```
python -m src.main -mode train -gpu 0 -use_attn -bidirectional -dataset quora -run_name <run_name>
```

### Create dictionary for submodular subset selection. Used for Semantic similarity (L<sub>2</sub>)
  
To use trained embeddings - 
```
python -m src.create_dict -model trained -run_name <run_name> -gpu 0
```

To use pretrained `word2vec` embeddings - 

```
python -m src.create_dict -model pretrained -run_name <run_name> -gpu 0
```

This will generate the `word2vec.pickle` file in `data/embeddings`

### Decoding using submodularity

```
python -m src.main -mode decode -selec submod -run_name <run_name> -beam_width 10 -gpu 0
```

### Citation

Please cite the following paper if you find this work relevant to your application

```tex
@inproceedings{dips2019,
    title = "Submodular Optimization-based Diverse Paraphrasing and its Effectiveness in Data Augmentation",
    author = "Kumar, Ashutosh  and
      Bhattamishra, Satwik  and
      Bhandari, Manik  and
      Talukdar, Partha",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1363",
    pages = "3609--3619"
}
```

For any clarification, comments, or suggestions please create an issue or contact [ashutosh@iisc.ac.in](http://ashutoshml.github.io) or [Satwik Bhattamishra](satwik55@gmail.com)
