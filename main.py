import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, render_template
from api.routes import routes

app = Flask(__name__)
app.register_blueprint(routes)

@app.route("/")
def home():
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
