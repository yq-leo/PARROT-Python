# PARROT-Python

Python implementation of [**"PARROT: Position-Aware Regularized Optimal Transport for Network Alignment"**](https://dl.acm.org/doi/abs/10.1145/3543507.3583357).
The official implementation is [here](https://github.com/zhichenz98/PARROT-WWW23).

### Prerequisites

- numpy
- scipy
- pytorch
- tqdm

### Datasets
You can run `main.py` using one of the following datasets

- ACM-DBLP-A
- ACM-DBLP-P
- cora
- foursquare-twitter
- phone-email

### Effiency
| Dataset / Runtime    | MATLAB            | PyTorch           |
|----------------------|--------------------|---------------------|
| **ACM_DBLP_A**       | 12.32s + 53.95s    | 110.55s + 241.52s   |
| **ACM_DBLP_P**       | 15.44s + 59.52s    | 89.29s + 241.11s    |
| **cora**             | 7.20s + 8.07s      | 3.11s + 7.78s       |
| **foursquare-twitter** | 6.89s + 17.59s    | 23.92s + 46.93s     |
| **phone-email**      | 0.32s + 1.18s      | 0.38s + 1.60s       |

### Usage

1. Clone the repository to your local machine:

```sh
git clone https://github.com/yq-leo/PARROT-Python.git
```

2. Navigate to the project directory:

```sh
cd PARROT-Python
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

4. To run PARROT, execute the following command in the terminal:
```sh
python main.py --dataset={dataset}
```

## Reference
### Official Code
[PARROT-WWW23](https://github.com/zhichenz98/PARROT-WWW23)

### Paper
Zeng, Z., Zhang, S., Xia, Y., & Tong, H. (2023, April). Parrot: Position-aware regularized optimal transport for network alignment. In Proceedings of the ACM Web Conference 2023 (pp. 372-382). [DOI](https://doi.org/10.1145/3543507.3583357).

