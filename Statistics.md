# Dataset Statitics 

| Dataset | Nodes | Edges | Features | Classes | Directed |Homo(edge) |Homo(node) | Assortativity | Avg. Degree | 
| :-----| ----: | ----: | ----: | ----: | :----: | ----: | ----: | ----: | ----: | 
| Cora         | 2708   | 5278     | 1433 | 7 | x | 0.810 | 0.825 | -0.066 | 3.898 |
| CiteSeer     | 3327   | 4552     | 3703 | 6 | x | 0.736 | 0.706 |  0.048 | 2.736 |
| PubMed       | 19717  | 44324    | 500  | 3 | x | 0.802 | 0.792 | -0.044 | 4.496 |
| Computers    | 13752  | 245861   | 767  | 10| x | 0.777 | 0.785 | -0.056 | 35.756| 
| Photo        | 7650   | 119081   | 745  | 8 | x | 0.827 | 0.836 | -0.045 | 31.132|
| Chameleon    | 2277   | 36101*   | 2277 | 10| x*| 0.235 | 0.104 | -0.113 | 31.709| 
| Squirrel     | 5201   | 217073*  | 2089 | 5 | x*| 0.224 | 0.089 |  0.374 | 83.474|
| Texas        |  183   |   325    | 1703 | 5 | x*| 0.108 | 0.065 | -0.346 | 1.776 |
| Cornell      |  183   |   298    | 1703 | 5 | x*| 0.305 | 0.212 | -0.384 | 1.628 |
| Wisconsin    |  251   |   515    | 1703 | 5 | x*| 0.196 | 0.172 | -0.272 | 2.052 |
| Actor        | 7600   | 30019    | 932  | 5 | x*| 0.219 | 0.159 | -0.111 | 3.950 | 
| Arxiv-year   | 169343 | 1166243  | 128  | 5 | o | 0.222 | 0.256 |  0.014 | 6.887 |
| Penn94       | 41554  | 1362229  | 4814 | 2 | x | 0.470 | 0.483 | -0.001 | 65.564| 
| Genius       | 421961 | 922868   | 12   | 2 | x*| 0.593 | 0.509 | -0.102 | 4.374 |
| Pokec        | 1632803| 22301964 | 65   | 2 | x*| 0.425 | 0.428 |  0.002 | 27.317|
| Twitch-gamer | 168114 | 6797557  |  7   | 2 | x*| 0.554 | 0.556 | -0.087 | 80.868|  
| Snap-patents | 2923922| 13975791 | 269  | 5 | o | 0.219 | 0.208 |  0.033 | 4.780 |
| Twitch-de    | 9498   | 153138   | 2514 | 2 | x*| 0.632 | 0.596 | -0.115 | 32.246|
| Deezer-Europe| 28281  | 92752    | 31241| 2 | x | 0.525 | 0.530 |  0.104 | 6.559 |
| Wiki         | 1925342| 303484860| 600  | 5 | x*| 0.378 | 0.258 | -0.015 |251.962|
| ogbn-arxiv   | 169343 | 1166243  |  128 | 40| o | 0.655 | 0.428 |  0.014 | 6.887 |
| ogbn-products|2449029 | 61859140 |  100 | 47| x | 0.808 | 0.817 | -0.042 | 50.517|
| ogbn-papers100M | 111059956 | 1615685872 | 128 | 172 | o | - | - | - | 14.548

## Note
* Undirected edges are counted only once. That is, there is only one edge between two connected edges in undirected graphs.
* Directed datasets are marked with 'o', undirected datasets are marked with 'x'. 
Some datasets are marked with 'x*' as they are naturally directed graphs, but converted to undirected version in common tasks. 
For these datasets, statistics like homophily is calculated based on the undirected version.
* Homo(edge) and Homo(node) diverse in the way they are calculated, known as edge homophily and node homophily.
* To support assortative calculation, one needs torch_geometric 2.2. 