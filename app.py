# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd

# -----------------------------
# 1. Initialize Flask app
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # change this to something secure
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# -----------------------------
# 2. Initialize DB & Migrate
# -----------------------------
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# -----------------------------
# 3. Database Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# -----------------------------
# 4. Load ML model and preprocessors
# -----------------------------
saved = joblib.load("best_crop_model.pkl")
model = saved["model"]
scaler = saved["scaler"]
ord_enc = saved["ordinal_encoder"]
categorical_cols = saved["categorical_cols"]
numerical_cols = saved["numerical_cols"]

# -----------------------------
# 5. Routes
# -----------------------------

# Home → requires login
@app.route("/")
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

# Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please login.", "warning")
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password)
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for('login'))

    return render_template("signup.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f"Welcome, {user.username}!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

# Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if 'user_id' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))

    try:
        # Extract inputs
        input_data = {col: [request.form.get(col)] for col in categorical_cols}
        input_data.update({col: [float(request.form.get(col))] for col in numerical_cols})

        df_input = pd.DataFrame(input_data)

        # Encode & scale
        df_input[categorical_cols] = ord_enc.transform(df_input[categorical_cols])
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

        prediction = model.predict(df_input)[0]
        return render_template("result.html", crop=prediction)

    except Exception as e:
        return f"❌ Error during prediction: {e}"

# -----------------------------
# 6. Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
