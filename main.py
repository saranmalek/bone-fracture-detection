import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, render_template
from api.routes import routes

app = Flask(__name__)
app.register_blueprint(routes)

@app.route("/")
def home():
    return render_template("index.html")

# DO NOT use app.run() when using Gunicorn
# Gunicorn will directly import and run the 'app' object

if __name__ == "__main__":
    # This only runs if you execute: python main.py
    # Gunicorn does NOT execute this
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
