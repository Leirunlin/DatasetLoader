from LINKX_dataset import LINKXDataset
import argparse
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon, Actor
from torch_geometric.data import Data, InMemoryDataset, download_url

root = "/data/runlin_lei/data"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='cora')
args = parser.parse_args()

name = args.dataset
name = name.lower()

### 
if name in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root=root, name=name)
elif name in ['Computers', 'Photo']:
    dataset = Amazon(root=root, name=name)
elif name in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(root=root, name=name)
elif name in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root=root, name=name)
elif name in ['film', 'actor']:
    root += '/actor/'
    dataset = Actor(root=root)
elif name in ['penn94', 'genius', 'wiki', 'pokec', 'arxiv-year',
              'twitch-gamer', 'snap-patents', 'twitch-de', 'deezer-europe']:
    dataset = LINKXDataset(root=root, name=name)

# test
data = dataset[0]
print(data)
print(dataset.num_classes)
