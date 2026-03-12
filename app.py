from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("fake_news_model.keras")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    if len(news) < 10:
        return render_template("index.html",
                               prediction="Please enter longer news text",
                               confidence="")

    # Convert input to TensorFlow dataset
    input_data = tf.data.Dataset.from_tensor_slices([news]).batch(1)

    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        result = "Real News ✅"
        confidence = round(prediction * 100, 2)
    else:
        result = "Fake News ❌"
        confidence = round((1 - prediction) * 100, 2)

    return render_template("index.html",
                           prediction=result,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)