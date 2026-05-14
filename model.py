from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


@app.route("/")
def home():

    return send_from_directory(".", "frontend.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        probability = 0.82

        return jsonify({

            "label": "Disease Detected",

            "confidence_percent":
                round(probability * 100, 2)

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )