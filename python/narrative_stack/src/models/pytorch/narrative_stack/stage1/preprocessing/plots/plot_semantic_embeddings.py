import matplotlib.pyplot as plt


def plot_semantic_embeddings(
    concept_unit_embeddings, labels=None, title="Embedding Scatterplot"
):
    """
    Visualize concept-unit embeddings in a 2D scatterplot.

    Parameters:
        concept_unit_embeddings (np.ndarray): Embedding matrix of shape (N, D),
            where D >= 2. Only the first 2 dimensions are used for plotting.
        labels (List[str], optional): Optional text labels to annotate each point.
        title (str): Title of the scatterplot.
    """

    # Select only the first 2 dimensions (or 3 if present) for plotting
    reshaped_embeddings = concept_unit_embeddings[:, :2]

    dim = reshaped_embeddings.shape[1]
    assert dim == 2, "Embeddings must be 2D for scatterplot"

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111)
    ax.scatter(reshaped_embeddings[:, 0], reshaped_embeddings[:, 1], s=10, alpha=0.7)

    if labels is not None:
        for i, label in enumerate(labels):
            ax.text(
                reshaped_embeddings[i, 0],
                reshaped_embeddings[i, 1],
                label,
                fontsize=6,
            )

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
