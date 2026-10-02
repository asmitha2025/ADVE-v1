"""
ADVE Vector PCA Compressor Module
Compresses 512-dimensional CLIP embeddings to 128-dimensional latent vectors.
Reduces FAISS index RAM/disk storage by 4x and speeds up vector retrieval latency.
"""

import numpy as np

class VectorCompressor:
    def __init__(self, target_dim=128):
        self.target_dim = target_dim
        self.projection_matrix = None
        self.mean_vector = None
        self.is_fitted = False

    def fit(self, embeddings):
        """Fit PCA projection matrix on sample embeddings (N x 512)"""
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.shape[0] < self.target_dim:
            # Random projection fallback if samples < 128
            np.random.seed(42)
            self.projection_matrix = np.random.randn(embeddings.shape[1], self.target_dim).astype(np.float32)
            self.projection_matrix /= np.linalg.norm(self.projection_matrix, axis=0, keepdims=True)
            self.mean_vector = np.zeros((1, embeddings.shape[1]), dtype=np.float32)
            self.is_fitted = True
            return

        self.mean_vector = np.mean(embeddings, axis=0, keepdims=True)
        centered = embeddings - self.mean_vector

        # SVD for PCA
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self.projection_matrix = vh[:self.target_dim, :].T
        self.is_fitted = True

    def compress(self, embedding):
        """Compress a single 512-dim embedding to 128-dim"""
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if not self.is_fitted:
            # Default truncation fallback
            compressed = emb[:, :self.target_dim]
            norm = np.linalg.norm(compressed)
            return (compressed / (norm + 1e-8)).flatten()

        centered = emb - self.mean_vector
        compressed = np.dot(centered, self.projection_matrix)
        norm = np.linalg.norm(compressed)
        return (compressed / (norm + 1e-8)).flatten()
