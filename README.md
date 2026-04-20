# 🎬 CineMatch: Hybrid Ensemble Movie Recommender

Created as a first year project under Basics Of Machine Learning Lab at DTU , CineMatch is a high-performance movie recommendation engine built from the ground up using **collaborative filtering** and **latent factor models**. The system features a unique 6-model ensemble architecture that integrates user-user similarity and genre embeddings, achieving a validation RMSE of **0.8647**—putting it within a 0.008 margin of the legendary BellKor’s Pragmatic Chaos (the 2009 Netflix Prize winner).Although , the amount of data used here is far less than what was in the actual one.

---

## 🚀 Key Features

* **6-Model Hybrid Architecture**: Ranges from baseline Biased SVD to a fully weighted ensemble of content and neighborhood models.
* **Dynamic Ensemble Learning**: A custom-weighted blending algorithm that adjusts model contributions based on real-time error backpropagation.(This was used first , but then a better accuracy was achieved with a different set of weights , so used that directly)
* **Contextual Genre Embeddings**: Enhances latent factor movie vectors with genre-specific weights.
* **Optimized Inference**: Utilizes a Top-20 Neighborhood Strategy for rapid similarity computation.
* **Live Web Interface**: A clean, responsive Flask-based UI for real-time movie discovery.

---

## 🏗 The Architecture

The system evaluates and blends six distinct algorithmic approaches:

| Model | Architecture | Logic |
| :--- | :--- | :--- |
| **Model 1** | **Biased SVD** | Traditional Matrix Factorization ($P_u \cdot Q_i + \text{biases}$) |
| **Model 2** | **Cosine Similarity** | Pure memory-based User-User Collaborative Filtering |
| **Model 3** | **SVD + Genre** | Augments movie vectors with learned genre embeddings |
| **Model 4** | **SVD + Cos Sim** | Augments user vectors with neighbor context vectors |
| **Model 5** | **Full Hybrid** | Combines Genre and Similarity augmentations |
| **Model 6** | **Weighted Ensemble** | A learned weighted sum of Models 1-5 |

---

## 📊 Performance Benchmarks

The model was trained using **Stochastic Gradient Descent (SGD)** with L2 regularization. 

| Metric | Ensemble | Model5 : SVD+Cosine +Genre | Model3 : SVD+Genre | Model4 : SVD+Cosine sim | Model1 : SVD | Model2 : Cosine similarity |
| :------- | :------- | :------- |:------- |:------- |:------- |:------- |
| **Test RMSE** | **0.8647** | **0.8656** | **0.8659** | **0.8660** | **0.8661** | **0.9743** |

-------

## 🛠 Tech Stack

* **Languages**: Python, HTML, CSS, JavaScript
* **Libraries**: NumPy, Pandas, Scikit-learn, Flask, Gunicorn
* **Deployment**: GitHub, Render (Cloud Hosting)

---

## 📂 Project Structure

```bash
├── app.py                # Flask Backend & Inference Engine
├── index.html            # Responsive UI (served via Flask)
├── model_params.npz      # Compressed Latent Vectors & Biases
├── model_misc.pkl        # Encoders & Similarity Mappings
├── movies.csv            # Movie Metadata
├── ratings_small.csv     # Pre-processed rating subset
└── requirements.txt      # Production dependencies
