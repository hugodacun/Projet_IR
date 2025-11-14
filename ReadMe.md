# Projet IR – Moteur de recherche & Dashboard d’évaluation

Ce projet implémente un petit moteur de recherche en Python sur un corpus de pages Wikipédia en français, avec :

- une interface de **recherche interactive** (`app.py`) ;
- une interface d’**évaluation des moteurs** (`app_eval.py`) pour comparer plusieurs méthodes (BM25, TF-IDF Cosine, hybrides, etc.).

L’interface est développée avec **Streamlit** et les graphes avec **Plotly**.

---

## Structure du projet

```text
IR_PROJECT/
├── app.py                # Interface Streamlit de recherche
├── app_eval.py           # Dashboard Streamlit d'évaluation / comparaison
├── requirements.txt      # Dépendances Python à installer via pip
├── README.md             
│
├── data/
│   ├── wiki_split_extract_2k/   # Corpus de documents (fichiers texte)
│   └── requetes.jsonl           # Fichier de requêtes + documents réponses (utilisé pour l’évaluation et la comparaison avec les résultats obtenus)
│
├── models/
│   ├── index.json        # Index inversé sauvegardé
│   └── edge_index.json   # Index pour l’auto-complétion (edge n-grams)
│
└── src/
│        ├── corpus.py #lecture du corpus de documents (fichiers texte) et fourniture des doc_id + contenu.
│        ├── index.py # Construction, sauvegarde et recherche dans l’index inversé (BM25, TF-IDF, hybrides).
│        ├── metrics.py # Calcul des métriques d’évaluation (MAP, P@K, nDCG, etc.)
│        ├── preprocess.py # Pré-traitement des textes (tokenization, stemming, etc.)
│        ├── search.py # Moteurs de recherche (BM25, TF-IDF, hybrides, etc.)
│        └── suggest.py # Moteur d’auto-complétion (edge n-grams)
│
└── test/
    └── test_preprocess.py   # Script de test pour vérifier le prétraitement sur les fichiers Wikipédia ( pas nécessaire pour l’utilisation des applications)

## Installation

### Prérequis

- Python 3.10+ recommandé
- `pip` à jour

---

### 1. Création de l’environnement virtuel

Depuis le dossier racine du projet (`IR_PROJECT`) :

```bash
python -m venv .venv
```

### Activation de l’environnement virtuel depuis la racine du projet
- Via le terminal de VS code :  `.\.venv\Scripts\activate.bat`

## 2. Installation des dépendances

Depuis l’environnement virtuel activé, exécuter :

```bash
pip install -r requirements.txt
```

## ▶Lancer les applications


### 1. Dashboard d'évaluation (`app_eval.py`)

Pour lancer le dashboard d évaluation :

```bash
python -m streamlit run app_eval.py
```

### 2. Interface de recherche (`app.py`)
⚠️Il faut obligatoirement lancer d’abord l’interface d’évaluation pour construire l’index inversé et les edge-ngrams pour la suggestion et le TF-IDF avant d’utiliser l’interface de recherche.

Pour lancer l’interface de recherche :

```bash
python -m streamlit run app.py
```

---
### Utilisation après ouverture de l’application

Une fois l’application ouverte :

- Ouvrir la sidebar (à gauche).

- Cliquer sur :
    - `Build/Rebuild index + edge n-grams`
    - puis `Build/Rebuild TF-IDF`.

- Aller dans l’onglet `Évaluation` ou `Comparaison`.

- Choisir :
    - la méthode : `BM25`, `TF-IDF Cosine`, `Hybrid RRF`, `Hybrid Interp` ;
    - les paramètres : `k`, `k_lex`, `k_vec`, `rrf_k`, `alpha`, etc.

- Cliquer sur `▶ Lancer l’évaluation` ou `▶ Lancer la comparaison`.

💡 Si l’index ou le TF-IDF ne sont pas prêts, une notification indique qu’il faut d’abord lancer les étapes de build dans la sidebar.
