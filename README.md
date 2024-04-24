# ZMGA: A ZINB-based Multi-modal Graph Autoencoder Enhancing Topological Consistency in Single-Cell Clustering
![Framework](https://github.com/cywang95/ZMGA/tree/main/fig/ZMGA.png)


**Brief introduction:**
ZMGA is a topologically consistent multi-modal graph autoencoder. Specifically, a Triple-graph Alignment module has been developed, utilizing compressed embeddings in the latent space to reconstruct graphs and ensure consistency in the topological structures of the reconstructed graphs with those of the topological graphs of each modality. Furthermore, to compress information effectively and model the distribution of real cell data accurately, we developed both a reconstruction module and a zero-inflated negative binomial (ZINB)  module.

**Packages:**
- Pandas==1.1.5
- Python==3.7.0
- torch==1.13.1
- NumPy==1.19.2
- SciPy==1.1.0
- Scikit-learn==0.19.0

**Datasets:**
We have released the dataset SNARE used in our paper as an example, the other datasets are introduced in maintext.

**Demo:**
You should change the code in load_data.py  

path = '/mnt/d/Code/ZMGA/data/' change to your path


**Implement:**
```python
python run_ZMGA.py
```



