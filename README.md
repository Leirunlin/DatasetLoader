# DatasetLoader
DatasetLoader based on pyG2.0 (NodeClassfication)

## Requirements
numpy==1.23.3  
ogb==1.3.4  
pandas==1.5.1  
scipy==1.8.1  
torch==1.12.1  
torch_geometric==2.1.0  

## Files
dataset_loader.py  
-- Load Dataset
LINKX_dataset.py
-- LINKX_dataset Loader

## DataStructure
Default Root: ```/data/runlin_lei/data```

FullStructure  
```
|-- root 
|  |-- actor (*)  
|  |-- arxiv-year
|  |  |-- (No raw files, store in ogbn-arixv)
|  |-- cora (*)
|  |-- citeseer (*) 
|  |-- Computers (*)
|  |-- cornell (*)
|  |-- genius
|  |  |-- raw
|  |  |  |-- data.mat (genius.mat, requires rename)
|  |  |  |-- genius-splits.npy (fixed splits)
|  |-- ogbn_arixv (downloaded from ogb)
|  |  |-- arxiv-year-splits.npy
|  |  |-- mapping
|  |  |-- processed
|  |  |-- raw
|  |  |-- split
|  |-- penn94
|  |  |-- raw
|  |  |  |-- data.mat (penn94.mat, requires rename)
|  |  |  |-- fb100-Penn94-splits.npy (fixed splits)
|  |-- Photo (*)
|  |-- pokec
|  |  |-- raw
|  |  |  |-- data.mat (pokec.mat, requires rename)
|  |  |  |-- pokec-splits.npy (fixed splits)
|  |-- pubmed
|  |-- snap-patents
|  |  |-- raw
|  |  |  |-- data.mat (snap-patents.mat, requires rename)
|  |  |  |-- snap-patents-splits.npy (fixed splits)
|  |-- squirrel (*)
|  |-- texas (*)
|  |-- twitch-de
|  |  |-- raw
|  |  |  |-- musae_de_edges.csv
|  |  |  |-- musae_de_features.json
|  |  |  |-- musae_de_target.csv
|  |  |  |-- twitch-e-de-splits.npy
|  |-- twitch-gamer
|  |  |-- raw
|  |  |  |-- twitch-gamer_edges.csv
|  |  |  |-- twitch-gamer_feat.json
|  |  |  |-- twitch-gamer-splits.npy
|  |-- wiki 
|  |  |-- raw
|  |  |  |-- wiki_edges2M.csv
|  |  |  |-- wiki_featuers2M.pt
|  |  |  |-- wiki_views2M.pt
|  |-- wisconsin (*)  
```
For datasets marked with (*), the raw datasets can be directly downloaded by pyG. 
If datasets are successfully downloaded, corresponding raw dataset will be included in folder "/raw.
For example, cora:
```
|  |-- cora 
|  |  |--raw
```

## Load Dataset
Running 
```
python dataset.py --dataset 'dataset_name'
```
to loader dataset.

For all datasets, after first running ```python dataset.py --dataset 'name' ```, processed dataset will be generated in folder "/processed".

For example, cora after processing:
```
|  |-- cora 
|  |  |--processed
|  |  |--raw
```

## Warning
* Directly download LINKX datasets and splits could fail. Please download raw files from [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) and store in the given format.
* Data splits are fixed following pyG except for wiki (fixed splits are not provided). For more flexible splits, please refer to [GPRGNN](https://github.com/jianhao2016/GPRGNN) and [BernNet](https://github.com/ivam-he/BernNet).



