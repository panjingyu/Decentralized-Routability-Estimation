# Routability Estimation base on Decentralized Private Data

## Data allocation & preprocessing

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/dataset/alloc_data.py
CUDA_VISIBLE_DEVICES=0 python scripts/dataset/preprocess_data.py
```

`alloc_data.man.py`: link raw data according to `designs.csv` under each client directory.
`alloc_data.py`: link raw data randomly under each client directory.
`preprocess_data.py`: resize & normalize raw data features to numpy files.
