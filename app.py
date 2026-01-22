from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

model = joblib.load("model/fake_news_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection</title>
    <style>
        body { font-family: Arial; background: #0f172a; color: white; padding: 40px; }
        textarea { width: 100%; height: 150px; }
        button { padding: 10px 20px; background: #22c55e; border: none; font-size: 16px; }
        .result { margin-top: 20px; font-size: 22px; }
    </style>
</head>
<body>
    <h1> Fake News Detection System</h1>

    <form method="post">
        <textarea name="news" placeholder="Paste news text here..." required></textarea><br><br>
        <button type="submit">Check News</button>
    </form>

    {% if result %}
        <div class="result">Result: <b>{{ result }}</b></div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        news_text = request.form["news"]
        prediction = model.predict([news_text])[0]
        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
