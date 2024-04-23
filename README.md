# PARROT-Python

Python implementation of [**"PARROT: Position-Aware Regularized Optimal Transport for Network Alignment"**](https://dl.acm.org/doi/abs/10.1145/3543507.3583357).
The official implementation is [here](https://github.com/zhichenz98/PARROT-WWW23).

### Prerequisites

- numpy
- scipy
- tqdm

### Datasets
You can run `main.py` using one of the following datasets

- ACM-DBLP-A
- ACM-DBLP-P
- cora
- foursquare-twitter
- phone-email

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

