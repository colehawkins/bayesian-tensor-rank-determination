This README assumes you have already loaded the [Criteo Kaggle dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) as described in the [Facebook DLRM repository](https://github.com/facebookresearch/dlrm). Most infrastructure and scripts in this repository are taken from the original DLRM repository.

The dataset should be located in the folder `input`.

## Setup

Install the appropriate additional requirements for this directory using

```
conda env update -f dlrm_reqs.yml
```
This will create the conda environment `dlrm_tensor`.
Then install our tensor layers package.
```
cd ../torch_bayesian_tensor_layers
pip install -e .
```
This should set up the tensorized layers package. The tensorized embedding package does not compute the full embedding, but instead only selects the necessary tensor slices. 

## Running training

The appropriate scripts are all located in the directory `bench`. Assuming you have set up a conda environment using the pip install above, you should be able to run the scripts in `bench` from the this directory (the `dlrm` directory).

Most script names are self-explanatory. The tensorized model scripts will print accuracy and ranks at each validation/test checkpoint. 

For example `run_cp_hc.sh` runs the half-cauchy CP model hyperparameter search.

In order to run the train-then-compress approach via `compress_then_train.sh` you will need to save a DLRM model.
