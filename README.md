# ZMGA: A ZINB-based Multi-modal Graph Autoencoder Enhancing Topological Consistency in Single-Cell Clustering
![Franework](https://github.com/cywang95/ZMGA/fig/ZMGA.pdf)
**Description:**

ZMGA is a topologically consistent multi-modal graph autoencoder. Specifically, a Triple-graph Alignment module has been developed, utilizing compressed embeddings in the latent space to reconstruct graphs and ensure consistency in the topological structures of the reconstructed graphs with those of the topological graphs of each modality. Furthermore, to compress information effectively and model the distribution of real cell data accurately, we developed both a reconstruction module and a zero-inflated negative binomial (ZINB)  module.

**Requirements:**


- Pandas==1.1.5
- pytorch==1.12.0
- NumPy==1.19.2
- SciPy==1.1.0
- Scikit-learn==0.19.0



**Examples:**

```python
parser.add_argument('--dataset_str', default='Biase', type=str, help='single cell dataset')
parser.add_argument('--n_clusters', default=3, type=int, help='number of clusters')


# Add other arguments as needed...


```
**Implement:**
```python
python ZMGA.py
```



