import faiss
import multiprocessing
import numpy as np
from tqdm import tqdm

class FaissNearestNeighbour:
    def __init__(self, num_workers=multiprocessing.cpu_count(), n_nearest_neighbours=1):
        faiss.omp_set_num_threads(num_workers)
        self.search_index = None
        self.n_nearest_neighbours = n_nearest_neighbours

    def train(self, features):
        if self.search_index:
            self.reset_index()
        self.search_index = self.create_index(features.shape[-1])
        self.search_index.add(features)

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

    def create_index(self, dimension):
        return faiss.IndexFlatL2(dimension)

    def save(self, output_file_path):
        faiss.write_index(self.search_index, output_file_path)

    def load(self, input_file_path):
        self.reset_index()
        self.search_index = faiss.read_index(input_file_path)

    def predict(self, features):
        query_distances, query_nns = self.search_index.search(features, self.n_nearest_neighbours)
        return query_distances