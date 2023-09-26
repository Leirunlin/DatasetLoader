# Dataset Statitics 

| Dataset | Nodes | Edges | Features | Classes | Directed |Homo(edge) |Homo(node) | Assortativity | Avg. Degree | 
| :-----| ----: | ----: | ----: | ----: | :----: | ----: | ----: | ----: | ----: | 
| Cora         | 2,708   | 5,278     | 1,433 | 7 | x | 0.810 | 0.825 | -0.066 | 3.898 |
| CiteSeer     | 3,327   | 4,552     | 3,703 | 6 | x | 0.736 | 0.706 |  0.048 | 2.736 |
| PubMed       | 19,717  | 44,324    | 500  | 3 | x | 0.802 | 0.792 | -0.044 | 4.496 |
| Computers    | 13,752  | 245,861   | 767  | 10| x | 0.777 | 0.785 | -0.056 | 35.756| 
| Photo        | 7,650   | 119,081   | 745  | 8 | x | 0.827 | 0.836 | -0.045 | 31.132|
| Chameleon    | 2,277   | 36,101*   | 2,277 | 5 | x*| 0.235 | 0.104 | -0.113 | 31.709| 
| Squirrel     | 5,201   | 217,073*  | 2,089 | 5 | x*| 0.224 | 0.089 |  0.374 | 83.474|
| Texas        |  183   |   325    | 1,703 | 5 | x*| 0.108 | 0.065 | -0.346 | 1.776 |
| Cornell      |  183   |   298    | 1,703 | 5 | x*| 0.305 | 0.212 | -0.384 | 1.628 |
| Wisconsin    |  251   |   515    | 1,703 | 5 | x*| 0.196 | 0.172 | -0.272 | 2.052 |
| Actor        | 7,600   | 30,019    | 932  | 5 | x*| 0.219 | 0.159 | -0.111 | 3.950 |
| Arxiv-year   | 169,343 | 1,166,243  | 128  | 5 | o | 0.222 | 0.256 |  0.014 | 6.887 |
| Penn94       | 41,554  | 1,362,229  | 4,814 | 2 | x | 0.470 | 0.483 | -0.001 | 65.564| 
| Genius       | 421,961 | 922,868   | 12   | 2 | x*| 0.593 | 0.509 | -0.102 | 4.374 |
| Pokec        | 1,632,803| 22,301,964 | 65   | 2 | x*| 0.425 | 0.428 |  0.002 | 27.317|
| Twitch-gamer | 168,114 | 6,797,557  |  7   | 2 | x*| 0.554 | 0.556 | -0.087 | 80.868|  
| Snap-patents | 2,923,922| 13,975,791 | 269  | 5 | o | 0.219 | 0.208 |  0.033 | 4.780 |
| Twitch-de    | 9,498   | 153,138   | 2,514 | 2 | x*| 0.632 | 0.596 | -0.115 | 32.246|
| Deezer-Europe| 28,281  | 92,752    | 31,241| 2 | x | 0.525 | 0.530 |  0.104 | 6.559 |
| Wiki         | 1,925,342| 303,484,860| 600  | 5 | x*| 0.378 | 0.258 | -0.015 |251.962|
| Roman_empire | 22,622 | 65,854 | 300 | 18 | x | 0.047 | 0.046 | -0.028 | 2.906 |
| Amazon_ratings | 24,492 | 186,100 | 300 | 5 | x | 0.380 | 0.376 | -0.092 | 7.598 |
| Minesweeper  | 10,000 | 78,804 | 7 | 2 | x | 0.683 | 0.683 | 0.392 | 7.880 |
| Questions    | 48,921 | 307,080 | 301 | 2 | x | 0.840 | 0.898 | -0.152 | 6.277 |
| Tolokers     | 11,758 | 1,038,000 | 10 | 2 | x | 0.595 | 0.634 | -0.080 | 88.280 |
| ogbn-arxiv   | 169,343 | 1,166,243  |  128 | 40| o | 0.655 | 0.428 |  0.014 | 6.887 |
| ogbn-products| 2,449,029 | 61,859,140 |  100 | 47| x | 0.808 | 0.817 | -0.042 | 50.517|
| ogbn-papers100M | 111,059,956 | 1,615,685,872 | 128 | 172 | o | - | - | - | 14.548

## Note
* Undirected edges are counted only once. That is, there is only one edge between two connected edges in undirected graphs.
* Directed datasets are marked with 'o', undirected datasets are marked with 'x'. 
Some datasets are marked with 'x*' as they are naturally directed graphs, but converted to undirected version in common tasks. 

For these datasets, statistics like homophily is calculated based on the undirected version.
* Homo(edge) and Homo(node) diverse in the way they are calculated, known as edge homophily and node homophily.
* To support assortative calculation, torch_geometric >= 2.2 is needed. 
