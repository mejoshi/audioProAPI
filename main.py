import os
import urllib.request
import librosa
from flask import Flask, request
from flask_restful import Api, Resource, reqparse, abort
import yo

app = Flask(__name__)
api = Api(app)

song_url = {}
song_url_args = reqparse.RequestParser()
song_url_args.add_argument("url", type=str, help="Need URL", required=True)


def signal_calculate(args):
    signal, sr = librosa.load(args)
    return sr


class audioProAPI(Resource):
    def get(self,song_id):
        # sr = signal_calculate(song_url[song_id]['url'])
        load = urllib.request.urlretrieve(song_url[song_id]['url'],"mysite/songwave.wav")
        signal, sr = librosa.load("mysite/songwave.wav")
        prediction = yo.predict("mysite/songwave.wav")
        if (sr == 22050):
            return {"prediction ": prediction}
        else:
            return {"sr" : "not found"}

    def put(self,song_id):
        args = song_url_args.parse_args()
        song_url[song_id] = args
        return {song_id:args}

api.add_resource(audioProAPI, "/audioProAPI/<int:song_id>")

if __name__ == '__main__':
    app.run(debug=True)

