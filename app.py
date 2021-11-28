import csv
import os

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_file

app = Flask(__name__)


@app.route("/")
def home():
    images = os.listdir(os.path.join(app.static_folder, "processed"))
    with open("detections.csv") as csvfile:
        detections = csv.reader(csvfile)
        lines = [line for line in detections]
    return render_template("home.html", data=zip(images, lines))


@app.route("/add", methods=['POST'])
def add():
    data = request.json
    with open("detections.csv") as csvfile:
        detections = csv.reader(csvfile)
        lines = [line for line in detections]
        id = int(lines[-1][0]) + 1 if len(lines) else 0

    with open("detections.csv", "a") as csvfile:
        csvfile.write(",".join(
            [str(id), str(data["small"]), str(data["medium"]), str(data["big"]), str(data["fly"])]))
        csvfile.write("\n")

    arr = np.array(data["processed"])
    arr = arr[:, :, ::-1]

    img = Image.fromarray(arr.astype('uint8'))
    img.save(f"static/processed/processed_{id}.jpg")
    return redirect(url_for('home'))


@app.route("/images/<id>", methods=['GET'])
def images(id):
    return send_file(f"static/processed/processed_{id}.jpg")


@app.route("/remove/<id>", methods=['GET'])
def remove(id):
    print(id)
    os.remove(f"static/processed/processed_{id}.jpg")
    with open("detections.csv") as csvfile:
        detections = csv.reader(csvfile)
        lines = [line for line in detections]

    with open("detections.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for line in lines:
            id_line = line[0]
            if id_line != id:
                writer.writerow(line)

    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run()
