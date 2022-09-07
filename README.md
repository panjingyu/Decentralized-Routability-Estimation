# Routability Estimation base on Decentralized Private Data

## Dataset

The routability dataset used in this experiment is available [here](https://www.dropbox.com/s/8fj7evl3vowjbz5/data_raw.7z?dl=1).
Under the project root directory, extract the compressed file with `7z x data_raw.7z`.

## Data allocation & preprocessing

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/dataset/alloc_data.py
CUDA_VISIBLE_DEVICES=0 python scripts/dataset/preprocess_data.py
```

`alloc_data.man.py`: link raw data according to `designs.csv` under each client directory.
`alloc_data.py`: link raw data randomly under each client directory.
`preprocess_data.py`: resize & normalize raw data features to numpy files.

## Training & testing

The scripts for training and testing are under `scripts/` directory.

For example, call
```bash
python scripts/train/fedprox.py [options]
```
to train a RouteNet model with default parameters. 
To see help, call
```bash
python scripts/train/fedprox.py --help
```

