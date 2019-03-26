from flask import Flask, jsonify, make_response, render_template
import json

from PaletteEmbeddingModel import PaletteEmbeddingModel
from PaletteSearchIndex import PaletteSearchIndex

import glob
data_files = glob.glob("./data/*.json")
data = []

for file in data_files:
    data.extend(json.load(open(file)))

model = PaletteEmbeddingModel()

palettes = list(map(lambda r: r["Palette"], data))
total_palettes = len(palettes)

model.BatchEmbed(palettes)

# Create an palette search index.
palette_index = PaletteSearchIndex(model, palettes)

print("{} palettes were processed".format(total_palettes))

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html", count=total_palettes, url="http://img.buzzfeed.com/buzzfeed-static/static/2013-10/enhanced/webdr06/15/9/anigif_enhanced-buzz-25158-1381844793-0.gif");

# test_data = [
#     {"id": "1", "imgUrl": "https://cs2.livemaster.ru/storage/ca/27/1286d2455b54c188e66afee5d1cz--ukrasheniya-dva-okeana-kulon-podveska-s-galkoj-i-morskim-stek.jpg"},
#     {"id": "2", "imgUrl": "https://cs2.livemaster.ru/storage/ca/27/1286d2455b54c188e66afee5d1cz--ukrasheniya-dva-okeana-kulon-podveska-s-galkoj-i-morskim-stek.jpg"},
#     {"id": "3", "imgUrl": "https://cs2.livemaster.ru/storage/ca/27/1286d2455b54c188e66afee5d1cz--ukrasheniya-dva-okeana-kulon-podveska-s-galkoj-i-morskim-stek.jpg"},
#     {"id": "4", "imgUrl": "https://cs2.livemaster.ru/storage/ca/27/1286d2455b54c188e66afee5d1cz--ukrasheniya-dva-okeana-kulon-podveska-s-galkoj-i-morskim-stek.jpg"},
#     {"id": "5", "imgUrl": "https://cs2.livemaster.ru/storage/ca/27/1286d2455b54c188e66afee5d1cz--ukrasheniya-dva-okeana-kulon-podveska-s-galkoj-i-morskim-stek.jpg"}
# ];
#
# @app.route("/tmp")
# def test():
#     return render_template("palettes.html", items=test_data);

@app.route("/palettes/<string:palette>")
def getPalettes(palette):
    palette_query = palette  # '141316-e6e7e8-617631-201f27-b4b79a'
    num_neighbors = 120

    result = []

    indices, distances = palette_index.GetNearestNeighbors(
        palette_query, num_neighbors)

#    for index, distance in zip(indices, distances):
#        result.append({"imgUrl": data[index]['ImgUrl'], "itemUrl": data[index]['ItemUrl'], "dist": distance})

#    return jsonify({"data": result});

    for index, distance in zip(indices, distances):
        result.append({"imgUrl": data[index]['ImgUrl'], "id": data[index]['ItemUrl'], "dist": distance})

    return render_template("palettes.html", items=result);


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
