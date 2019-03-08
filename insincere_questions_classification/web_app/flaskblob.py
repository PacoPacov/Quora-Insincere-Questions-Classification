from flask import Flask, render_template
import os
app = Flask(__name__)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(project_dir, 'models')
models = [{'file_name': f} for f in os.listdir(models_path) 
            if (f.endswith('.pickle') or f.endswith('.pkl')) and "w2v" not in f]


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", models=models)

if __name__ == "__main__":
    print (__file__)
    app.run(debug=True)
