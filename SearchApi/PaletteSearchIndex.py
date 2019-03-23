import annoy

from ModelParameters import EMBEDDING_DIMENSION, NUM_ANNOY_TREES


class PaletteSearchIndex(object):
  """Data structure for nearest-neighbor search in color palette space.

  Attributes:
    _embedding_model: PaletteEmbeddingModel.
    _index: annoy.AnnoyIndex.
  """

  def __init__(self, embedding_model, palettes):
    """Inits PaletteSearchIndex with a list of color palettes.

    The palette index will compute the nearest neighbors in the input palette
    list.

    Args:
      embedding_model: PaletteEmbeddingModel. The embedding model to use.
      palettes: A list of strings each representing a 5-color palette.
    """
    self._embedding_model = embedding_model
    embeddings = self._embedding_model.BatchEmbed(palettes)
    self._index = annoy.AnnoyIndex(EMBEDDING_DIMENSION, metric='euclidean')
    for i, embedding in enumerate(embeddings):
      self._index.add_item(i, embedding)
    self._index.build(NUM_ANNOY_TREES)

  def GetNearestNeighbors(self, palette, num_neighbors):
    """Return the nearest neighbors of the input palette.

    Will return the index of the nearest palettes in the palette list that was
    used to initialize the PaletteSearchIndex.

    Args:
      palette: A string representing a color palette.
      num_neighbors: The number of neighbors to return.

    Returns:
      A pair of lists. The first list contains the indices of the neighbors. The
      second one contains the distances to the neighbors. Both list are sorted
      by neighbor distance.
    """
    embedding = self._embedding_model.Embed(palette)
    return self._index.get_nns_by_vector(
        embedding, num_neighbors, include_distances=True)