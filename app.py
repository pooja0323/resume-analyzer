from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import PyPDF2
import spacy

# ML imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("App starting...")

app = Flask(__name__)
app.secret_key = "secret123"

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Load NLP
print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded!")

# ------------------ MODELS ------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Float)
    missing_skills = db.Column(db.String(500))
    user_id = db.Column(db.Integer)

# ------------------ LOGIN ------------------

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------ FUNCTIONS ------------------

def extract_text(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# 🔥 ML BASED ANALYSIS
def analyze_resume(resume_text, job_desc):
    if not resume_text or not job_desc:
        return 0, []

    # ML similarity
    cv = CountVectorizer()
    matrix = cv.fit_transform([resume_text, job_desc])

    similarity = cosine_similarity(matrix)[0][1]
    score = round(similarity * 100, 2)

    # Missing skills
    resume_words = set(resume_text.lower().split())
    job_words = set(job_desc.lower().split())

    missing = job_words - resume_words

    return score, list(missing)[:10]

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return redirect(url_for("login"))

# -------- REGISTER --------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

# -------- LOGIN --------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            login_user(user)
            return redirect(url_for("index"))
        else:
            return "Invalid Credentials"

    return render_template("login.html")

# -------- RESUME ANALYZER --------
@app.route("/index", methods=["GET","POST"])
@login_required
def index():
    score = None
    missing = []

    if request.method == "POST":
        pdf = request.files["resume"]
        job_desc = request.form["job_desc"]

        resume_text = extract_text(pdf)
        score, missing = analyze_resume(resume_text, job_desc)

        # SAVE RESULT
        result = Result(
            score=score,
            missing_skills=",".join(missing),
            user_id=current_user.id
        )
        db.session.add(result)
        db.session.commit()

    return render_template("index.html", score=score, missing=missing, user=current_user.username)

# -------- HISTORY --------
@app.route("/history")
@login_required
def history():
    results = Result.query.filter_by(user_id=current_user.id).all()
    return render_template("history.html", results=results)

# -------- LOGOUT --------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ------------------ RUN ------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    print("Running Flask...")
    app.run(debug=True)
