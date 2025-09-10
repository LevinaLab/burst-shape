from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, Flask!"


if __name__ == "__main__":
    app.run(debug=False, port=8050)
    # app.run(DEBUG=False, port=5000, host="0.0.0.0")
