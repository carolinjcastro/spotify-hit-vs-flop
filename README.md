# Spotify Hit vs Flop

Spotify Hit vs Flop is an interactive dashboard for exploring how musical features influence a track’s potential to become a hit. Built using Python, Dash, Plotly, and Statsmodels, this project predicts the hit likelihood of any given song and helps visualize its placement in the broader music feature landscape.

Using the dashboard, users can:
- Predict the probability of a song becoming a hit based on its audio features
- Compare hit and flop distributions for features like danceability, energy, valence, and more
- Explore logistic regression results that drive the model
- Visually locate a song's position in feature space compared to thousands of other tracks

---

## Demo

You’ll see your song highlighted with a ★ purple star.

![](assets/demo.mp4)

---

## Getting Started

This app was developed locally and is currently view-only. A hosted public version may be added in the future. For now, this repository includes all code and documentation used for the project.

---

## Features & Workflow

- Cleaned a large Spotify dataset (20k+ tracks) using `pandas`
- Built and trained a logistic regression model using `statsmodels`
- Designed an interactive layout using `Dash` and `Plotly`
- Visualized regression coefficients, correlation trends, and scatter/jitter plots
- Highlighted user-selected tracks within distributions for interpretability

---

## File Structure
SpotifyDashApp/
├── app.py                  # Main Dash application code

├── Spotify_dataset.csv     # Raw dataset used for training and visualization

├── assets/                 # (Optional) Add custom stylesheets or images here

├── README.md               # Project overview and documentation

Note: The dataset is loaded as-is and cleaned within the app.py script.
