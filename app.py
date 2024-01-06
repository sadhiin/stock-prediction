from flask import Flask, request, jsonify, render_template
import os
import yaml
import numpy as np
import joblib
from prediction_service import prediction
from flask_cors import cross_origin
from src.utils import read_params, logger


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
# @cross_origin()
def index():
    if request.method == "POST":
        try:
            if request.form:
                data_req = dict(request.form).values()
                print(data_req)
                data_req = list(map(float, data_req))
                response = prediction(data_req)
                print(response)
                return render_template("index.html", response=response)
            else:
                return render_template("404.html")
        except Exception as e:
            logger.error(f"Something went wrong: {e}")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
