from time import time

from sklearn.manifold import TSNE

embeddings_dict = {
    't-SNE embedding': TSNE(
        n_components=2,
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    )
}

def get_plot_features(X, y, embedding_name='t-SNE embedding', return_time=False):
    if embedding_name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X
    print(f"Computing {embedding_name}...")
    start_time = time()
    projection = embeddings_dict[embedding_name].fit_transform(data, y)
    timing = time() - start_time
    if return_time:
        return projection, timing
    else:
        return projection