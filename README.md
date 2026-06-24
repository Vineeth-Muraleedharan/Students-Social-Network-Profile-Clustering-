# 🎓 Students' Social Network Profile Clustering

**Course:** Applied Data Science, ML & AI
**Institute:** E&ICT Academy, IIT Guwahati

**App Link :** https://students-social-network-profile-clustering-app.streamlit.app/
---

## 📌 Project Overview

This project segments high school students based on their **social network profile keywords** using unsupervised machine learning. The dataset contains 15,000 student profiles with 36 keyword mention counts across interest themes like sports, music, religion, fashion, romance, and risky behaviour - along with demographic features like age, gender, graduation year, and number of friends.

**Goal:** Identify natural student groupings based on similar interests, demographic profiles, and trends over graduation year using three clustering algorithms - KMeans, Hierarchical, and DBSCAN.

---

## 📂 Repository Structure

```
├── app.py                        # Streamlit web app
├── code.ipynb                    # Full analysis notebook
├── Clustering_Marketing.csv      # Dataset
└── requirements.txt              # Python dependencies
```

---

## 📊 Dataset

- **Source:** [Kaggle - Students Social Network Profile Clustering](https://www.kaggle.com/datasets/zabihullah18/students-social-network-profile-clustering/data)
- **Records:** 15,000 student profiles
- **Features:** 40 columns - 4 demographic + 36 keyword interest columns

| Theme | Keywords |
|---|---|
| ⚽ Sports | basketball, football, soccer, softball, volleyball, swimming, cheerleading, baseball, tennis, sports |
| 🎵 Music/Band | band, marching, music, rock, dance |
| ✝️ Religion | god, church, jesus, bible |
| 💅 Appearance | cute, hair, dress, blonde, mall, shopping, clothes, hollister, abercrombie |
| 💋 Romance | sex, sexy, hot, kissed |
| 💀 Risky | die, death, drunk, drugs |

---

## 📓 Notebook - `code.ipynb`

Full step-by-step analysis:

| Question | Description |
|---|---|
| **i** | Load dataset - shape, dtypes, missing values |
| **ii** | EDA - demographic distributions, keyword frequency, correlation heatmap, gender breakdown, trend over graduation year |
| **iii** | Skewness check, log1p transformation on high-skew features |
| **iv** | IQR outlier detection, Winsorization, gender encoding, StandardScaler |
| **v** | PCA dimensionality reduction, KMeans clustering, interest heatmap, demographic profile, trend analysis |
| **vi** | KMeans vs Hierarchical vs DBSCAN - Silhouette, Davies-Bouldin, Calinski-Harabasz comparison |

---

## 🤖 Streamlit App - `app.py`

Interactive web app with **4 tabs:**

| Tab | Description |
|---|---|
| 📊 Data Overview | Dataset summary, descriptive stats, keyword themes |
| 🤖 Clustering Results | PCA, optimal K (elbow + silhouette), 2D scatter, interest heatmap, theme profile, demographic profile, trend analysis, download CSV |
| 🏆 Method Comparison | Hierarchical (3 linkages + dendrogram), DBSCAN (k-distance + grid search), side-by-side scatter, final metric comparison table + bar charts |
| 🔮 Predict New Student | Enter student profile → instantly assigned to the nearest cluster with full cluster profile shown |

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Predict New Student Feature

Enter a new student's details:
- **Graduation Year** - dropdown (2006–2009)
- **Gender** - radio button (F / M)
- **Age** - number input (13–20)
- **Number of Friends** - number input
- **Theme Interest Levels** - select slider per theme (None → Low → Moderate → High → Very High)

Click **Assign to Cluster** → the app shows:
- Which cluster the student belongs to
- Cluster size, avg age, avg friends, % female
- Dominant theme interest profile
- Top 10 keywords for that cluster

---

## 🧪 Clustering Methods & Evaluation

| Method | Description |
|---|---|
| **KMeans** | Optimal K selected via elbow + silhouette score |
| **Hierarchical** | Ward, complete, average linkages compared |
| **DBSCAN** | Epsilon tuned via K-distance plot |

| Metric | Direction | Meaning |
|---|---|---|
| Silhouette Score | ↑ Higher is better | Measures cluster separation and cohesion |
| Davies-Bouldin | ↓ Lower is better | Measures compactness and separation |
| Calinski-Harabasz | ↑ Higher is better | Ratio of between-cluster to within-cluster variance |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn, scipy |
| Web App | Streamlit |

---

## ✍️ Author

**Vineeth Muraleedharan**
Senior Radiation Therapist | Healthcare AI Developer
E&ICT Academy, IIT Guwahati - Applied Data Science, ML & AI
