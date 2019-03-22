from flask import Flask, jsonify, make_response
import json

import annoy
import numpy as np
from skimage import color
import tensorflow as tf

# Parameters of the embedding model.
MODEL_DIR = './model'
TAG = tf.saved_model.tag_constants.SERVING
SIGNATURE_KEY = (
    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
IN_TENSOR_KEY = tf.saved_model.signature_constants.PREDICT_INPUTS
OUT_TENSOR_KEY = tf.saved_model.signature_constants.PREDICT_OUTPUTS

# Parameters of the palette search index.
EMBEDDING_DIMENSION = 15
NUM_ANNOY_TREES = 10

def RgbFromHex(color_hex):
  """Returns a RGB color from a color hex.

  Args:
    color_hex: A string encoding a single color. Example: '8f7358'.

  Returns:
    A RGB color i.e. a 3-int tuple. Example: (143, 115, 88).
  """
  return tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))


def GetPaletteFromString(palette_string):
  """Converts a string to a RGB color palette.

  Args:
    palette_string: A string encoding a color palette with color hexes. The
      expected format is 'color1-color2-color3-color4-color5' with colors
      encoded as hex strings. Example: '8f7358-8e463d-d4d1cc-26211f-f2f0f3'.

  Returns:
    A RGB color palette i.e. a list of RGB colors.
  """
  return [RgbFromHex(color_hex) for color_hex in palette_string.split('-')]


def ConvertPalettesToLab(rgb_palettes):
  """Converts a list of RGB color palettes to the Lab color space.

  Args:
    rgb_palettes: A list of RGB palettes.

  Returns:
    A list of Lab palettes. Lab palettes are a list of Lab colors i.e. a list of
    3-int tuples
  """
  temp = np.array(rgb_palettes)
  scaled_palettes = temp / 255.0
  return color.rgb2lab(scaled_palettes)


class PaletteEmbeddingModel(object):
  """Runs a palette embedding model.

  The Euclidean distance between two palette embeddings is a perceptual distance
  between the original palettes. Supports only 5-color palettes. Will produce
  15-dimensional dense embeddings.

  Attributes:
    _sess: A Tensorflow session running the embedding model.
    _in_tensor: The input tensor of the model. Should be fed a Lab color
    palette.
    _out_tensor: The output tensor of the model. Will contain the palette
    embedding.
  """

  def __init__(self):
    """Inits PaletteEmbeddingModel with the demo saved model."""

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1

    self._sess = tf.Session(graph=tf.Graph(), config=config)
    meta_graph_def = tf.saved_model.loader.load(self._sess, [TAG], MODEL_DIR)
    signature = meta_graph_def.signature_def
    in_tensor_name = signature[SIGNATURE_KEY].inputs[IN_TENSOR_KEY].name
    out_tensor_name = signature[SIGNATURE_KEY].outputs[OUT_TENSOR_KEY].name
    self._in_tensor = self._sess.graph.get_tensor_by_name(in_tensor_name)
    self._out_tensor = self._sess.graph.get_tensor_by_name(out_tensor_name)

  def BatchEmbed(self, palettes):
    """Returns the embedding of a list of color palettes.

    Args:
      palettes: A list of strings each representing a 5-color palette.

    Returns:
      A list of 15-D numpy arrays. The size of the list is equal to the size of
      the input palette list.
    """
    rgb_palettes = [GetPaletteFromString(palette) for palette in palettes]
    lab_palettes = ConvertPalettesToLab(rgb_palettes)
    in_tensors = [lab_palette.flatten() for lab_palette in lab_palettes]
    return self._sess.run(self._out_tensor, {self._in_tensor: in_tensors})

  def Embed(self, palette):
    """Returns the embedding of a single color palette.

    Args:
      palette: A string representing a 5-color palette.

    Returns:
      A 15-D numpy array.
    """
    return self.BatchEmbed([palette])[0]

  def ComputeDistance(self, a, b):
    """Returns a perceptual distance between two palettes.

    Args:
      a: A palette (string).
      b: Another palette (string).

    Returns:
      The distance between the palettes as a float.
    """
    embeddings = self.BatchEmbed([a, b])
    return np.linalg.norm(embeddings[0] - embeddings[1])


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


model = PaletteEmbeddingModel()

# Compute an abstract embedding of some sample palettes.
# embeddings = model.BatchEmbed(SAMPLE_PALETTES)

with open("./data/posuda.kruzhki-i-chashki.palette.json") as f:
  data = json.load(f)

with open("./data/odezhda.yubki.palette.json") as f:
  data = data + json.load(f)

with open("./data/odezhda.platya.palette.json") as f:
 data = data + json.load(f)

with open("./data/kosmetika-ruchnoj-raboty.palette.json") as f:
 data = data + json.load(f)

with open("./data/otkrytki.palette.json") as f:
 data = data + json.load(f)

with open("./data/russkij-stil.palette.json") as f:
 data = data + json.load(f)

with open("./data/muzykalnye-instrumenty.palette.json") as f:
 data = data + json.load(f)

with open("./data/kantselyarskie-tovary.palette.json") as f:
 data = data + json.load(f)

with open("./data/sumki-i-aksessuary.palette.json") as f:
 data = data + json.load(f)

with open("./data/tsvety-i-floristika.palette.json") as f:
 data = data + json.load(f)

palettes = list(map(lambda r: r["Palette"], data))

model.BatchEmbed(palettes)

# Create an palette search index.
palette_index = PaletteSearchIndex(model, palettes)

print ("{} palettes were processed".format(len(palettes)))

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/palettes/<string:palette>")
def getPalettes(palette):
    palette_query = palette #'141316-e6e7e8-617631-201f27-b4b79a'
    num_neighbors = 40

    result = []

    indices, distances = palette_index.GetNearestNeighbors(
        palette_query, num_neighbors)

    for index, distance in zip(indices, distances):
        result.append({"imgUrl": data[index]['ImgUrl'], "itemUrl": data[index]['ItemUrl'], "dist": distance})

    return jsonify({"data": result});

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
