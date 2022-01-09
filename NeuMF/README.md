## Neural Collaborative Filtering, NeuMF

NeuMF is a combination of matrix factorization and multilayer perceptron. The multilayer perceptron takes the concatenation of user and item embeddings as the input.

<p align="center">
<img src="./NeuMF/neumf.png" width="600" height="400" />
</p>


In this repository you'll find the tensorflow-recommender implementation for NeuMF. To test the architecture for your own case:

- create your own embedding models for item and user.
- in main.py, change the initialization of the embedddings for both MF and MLP.
