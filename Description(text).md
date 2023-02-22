# Description for Included Datasets

This file contains a brief description of included datasets including both Chineses and English versions.
The description mostly refers the dataset cheatsheat in PyG [link](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html) and original papers.

## Plantoid
This dataset include three citation network *Cora*, *CiteSeer* and *PubMed*.
These networks are first used in transductive node classification in ["Revisiting Semi-Supervised Learning with Graph Embeddings"](https://arxiv.org/abs/1603.08861) paper. 

In these datasets, nodes represent documents and edges represent citation links. 
Node features are bag-of-words representation of documents, class labels are given by the category. 

## Amazon
This dataset includes two sub-datasets  *Computers* and *Photos* and is proposed in paper ["Pitfalls of Graph Neural Network Evaluation"](https://arxiv.org/abs/1811.05868).

Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph, which is proposed in ["Image-based Recommendations on Styles and Substitutes"](https://arxiv.org/abs/1506.04757), where nodes represent goods, edges indicate that two goods are frequently bought together, node features are bag-of-words encoded product reviews, and class labels are given by the product category.

## Wikipedia
This dataset include two sub-datasets *Chameleon* and *Squirrel*. 
The Wikipedia networks are introduced in the [“Multi-scale Attributed Node Embedding”](https://arxiv.org/abs/1909.13021) paper. 

Nodes represent web pages and edges represent hyperlinks between them. 
Node features represent several informative nouns in the Wikipedia pages. 
The task is to predict the average daily traffic of the web page.

## WebKB and Actor
This dataset include four sub-datasets used in ["Geom-GCN: Geometric Graph Convolutional Networks"](https://openreview.net/forum?id=S1e2agrFvS) which are *Cornell*, *Texas*, *Wisconsin* and *Actor*.  

For *Cornell*, *Texas*, *Wisconsin*, nodes represent web pages and edges represent hyperlinks between them. 
Node features are the bag-of-words representation of web pages. The task is to classify the nodes into one of the five categories, student, project, course, staff, and faculty.

For *Actor*, each node corresponds to an actor, and the edge between two nodes denotes co-occurrence on the same Wikipedia page. 
Node features correspond to some keywords in the Wikipedia pages. 
The task is to classify the nodes into five categories in term of words of actor’s Wikipedia.

## LINKX dataset
This dataset include sub-datasets proposed in [“Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods”](https://arxiv.org/abs/2110.14446), including *Penn94*, *Pokec*, *Arxiv-year*, *snap-patents*, *twitch-gamers* and *wiki*.
The following description is taken from the original paper.

Penn94 from paper ["Social structure of facebook networks"](https://arxiv.org/pdf/1102.2166.pdf) is a friendship network from the Facebook 100 networks of university students from 2005, where nodes represent students. 
Each node is labeled with the reported gender of the user. 
The node features are major, second major/minor, dorm/house, year, and high school. 

Pokec from paper ["Stanford large network dataset collection"](https://snap.stanford.edu) is the friendship graph of a Slovak online social network, where nodes are users and edges are directed friendship relations. Nodes are labeled with reported gender. 
Node features are derived from profile information, such as geographical region, registration time, and age. 

ArXiv-year from paper ["Open graph benchmark: Datasets for machine learning on graphs"](https://papers.neurips.cc/paper/2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-Paper.pdf) is the ogbn-arXiv network with different labels. Our contribution is to set the class labels to be the year that the paper is posted, instead of paper subject area. The nodes are arXiv papers, and directed edges connect a paper to other papers that it cites. 
The node features are averaged word2vec token features of both the title and abstract of the paper. 
The five classes are chosen by partitioning the posting dates so that class ratios are approximately balanced.

Snap-patents from paper ["Open graph benchmark: Datasets for machine learning on graphs"](https://papers.neurips.cc/paper/2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-Paper.pdf) and ["Stanford large network dataset collection"](https://snap.stanford.edu) is a dataset of utility patents in the US. 
Each node is a patent, and edges connect patents that cite each other. Node features are derived from patent metadata. 
The task is to predict the time at which a patent was granted, resulting in five classes. 

Twitch-gamers from paper ["Twitch gamers: a dataset for evaluating proximity preserving and structural role-based node embeddings."](https://arxiv.org/abs/2101.03091) is a connected undirected graph of relationships between accounts on the streaming platform Twitch. 
Each node is a Twitch account, and edges exist between accounts that are mutual followers. 
The node features include number of views, creation and update dates, language, life time, and whether the account is dead. 
The binary classification task is to predict whether the channel has explicit content. 

Wiki is a dataset of Wikipedia articles, where nodes represent pages and edges represent links between them. 
Node features are constructed using averaged title and abstract GloVe embeddings. 
Labels represent total page views over 60 days, which are partitioned into quintiles to make five classes.

## OGB
The ogbn-arxiv and ogbn-products dataset from ["Open graph benchmark: Datasets for machine learning on graphs"](https://papers.neurips.cc/paper/2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-Paper.pdf)

Ogbn-arxiv is a directed graph, representing the citation network between all Computer Science (CS) arXiv papers indexed by MAG. 
Each node is an arXiv paper and each directed edge indicates that one paper cites another one. 
Each paper comes with a 128-dimensional feature vector obtained by averaging the embeddings of words in its title and abstract. 
The embeddings of individual words are computed by running the skip-gram model over the MAG corpus. 
The task is to predict the 40 subject areas of arXiv CS papers, e.g., cs.AI, cs.LG, and cs.OS, which are manually determined (i.e., labeled) by the paper’s authors and arXiv moderators. Formally, the task is to predict the primary categories of the arXiv papers, which is formulated as a 40-class classification problem.
The authors consider a realistic data split based on the publication dates of the papers. 
Specifically, they propose to train on papers published until 2017, validate on those published in 2018, and test on those published since 2019.

Ogbn-products is an undirected and unweighted graph, representing an Amazon product co-purchasing network. 
Nodes represent products sold in Amazon, and edges between two products indicate that the products are purchased together. The graphs, target labels, and node features are generated from bag-of-words of the product descriptions. 
The task is to predict the category of a product in a multi-class classification setup, where the 47 top-level categories are used for target labels.
The authors consider a more challenging and realistic dataset splitting. 
They use the sales ranking (popularity) to split nodes into training/validation/test sets. Specifically, they sort the products according to their sales ranking and use the top 8% for training, next top 2% for validation, and the rest for testing. 
