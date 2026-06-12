# Projets AI & Big Data

## Objectif
Chaque projet doit être un système complet incluant :
- Data pipeline
- Modèle AI
- API
- Application (Web / Mobile / Desktop)
- Déploiement (Docker)

---

## 1. Système de détection d’anomalies en streaming (fraude temps réel)

* **Type** : Anomaly Detection + Streaming (Spark)
* **Description** : Détecter des anomalies dans des flux de transactions en temps réel
* **Dataset** :
  [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Technos** : Spark Streaming, Kafka, Python, FastAPI
* **Application** : Web dashboard temps réel
* **Livrable attendu** :

  * Pipeline Kafka → Spark → Model
  * API scoring
  * Dashboard avec alertes live

---

## 2. Moteur de recommandation scalable (Big Data)

* **Type** : Recommandation + Spark ML
* **Dataset** :
  [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
* **Technos** : PySpark MLlib, Hadoop, Flask
* **Application** : Web app (Netflix-like)
* **Attendu** :

  * ALS distribué
  * API recommandation
  * Interface utilisateur

---

## 3. Segmentation client avec clustering distribué

* **Type** : Clustering (KMeans, DBSCAN)
* **Dataset** :
  [https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
* **Technos** : Spark ML, Plotly
* **App** : Dashboard interactif
* **Attendu** :

  * Clusters interprétables
  * Visualisation dynamique

---

## 4. Data Lake intelligent pour logs (Hadoop)

* **Type** : Big Data Engineering + NLP
* **Dataset** :
  [https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2018-developer-survey](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2018-developer-survey)
* **Technos** : Hadoop HDFS, Hive, Spark
* **App** : Interface de recherche
* **Attendu** :

  * Data lake structuré
  * Moteur de requête intelligent

---

## 5. Système de prédiction trafic routier temps réel

* **Type** : Time Series + Streaming
* **Dataset** :
  [https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)
* **Technos** : LSTM + Spark Streaming
* **App** : Web map
* **Attendu** :

  * Prédiction live
  * Visualisation carte

---

## 6. Détection de fake news à grande échelle

* **Type** : NLP + Classification
* **Dataset** :
  [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **Technos** : BERT + Spark NLP
* **App** : Web checker (coller un article)
* **Attendu** :

  * API NLP
  * Score crédibilité

---

## 7. Système de monitoring intelligent (DevOps AI)

* **Type** : Anomaly detection
* **Dataset** :
  [https://github.com/NetManAIOps/KPI-Anomaly-Detection](https://github.com/NetManAIOps/KPI-Anomaly-Detection)
* **Technos** : Prometheus + Spark + ML
* **App** : Dashboard DevOps
* **Attendu** :

  * Alerting automatique

---

## 8. Autoencoder pour compression de données Big Data

* **Type** : Autoencoder
* **Dataset** :
  [https://www.kaggle.com/datasets/crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)
* **Technos** : PyTorch + Spark
* **App** : Interface upload/compression
* **Attendu** :

  * Comparaison compression classique vs DL

---

## 9. Système de détection de bots sur réseaux sociaux

* **Type** : Classification + Graph
* **Dataset** :
  [https://www.kaggle.com/datasets/ashishjangra27/twitter-bot-detection](https://www.kaggle.com/datasets/ashishjangra27/twitter-bot-detection)
* **Technos** : Graph ML + Spark GraphX
* **App** : Dashboard analyse comptes
* **Attendu** :

  * Score bot humain

---

## 10. Système de pricing dynamique (e-commerce)

* **Type** : RL (Reinforcement Learning)
* **Dataset** :
  [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
* **Technos** : RL (DQN), Python
* **App** : Simulateur web
* **Attendu** :

  * Agent qui ajuste prix

---

## 11. Plateforme d’analyse de sentiments multi-langues (Afrique focus)

* **Type** : NLP
* **Dataset** :
  [https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
* **Technos** : Transformers
* **App** : Web app analyse texte
* **Attendu** :

  * Support FR/EN

---

## 12. Système de détection de pannes industrielles

* **Type** : Predictive Maintenance
* **Dataset** :
  [https://www.kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
* **Technos** : LSTM + Spark
* **App** : Dashboard industriel
* **Attendu** :

  * Prédiction panne

---

## 13. Système de recherche sémantique (mini Google)

* **Type** : NLP + Embeddings
* **Dataset** :
  [https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
* **Technos** : FAISS, BERT
* **App** : moteur de recherche web
* **Attendu** :

  * Recherche par sens (pas mot clé)

---

## 14. Détection d’objets temps réel (edge + cloud)

* **Type** : Computer Vision
* **Dataset** :
  [https://cocodataset.org/](https://cocodataset.org/)
* **Technos** : YOLO + API
* **App** : Webcam app
* **Attendu** :

  * Détection live

---

## 15. Système de scoring crédit alternatif (Afrique)

* **Type** : Classification + Feature engineering
* **Dataset** :
  [https://www.kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk)
* **Technos** : XGBoost + Spark
* **App** : App web scoring client
* **Attendu** :

  * Score crédit explicable

---

## 16. Chatbot intelligent basé sur base documentaire

* **Type** : NLP + RAG
* **Dataset** :
  Wikipedia dump / docs custom
* **Technos** : LangChain + vector DB
* **App** : Chat web
* **Attendu** :

  * QA sur documents

---

## 17. Système de détection de fraude télécom (CDR)

* **Type** : Big Data + anomaly detection
* **Dataset** :
  [https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
* **Technos** : Spark + ML
* **App** : Dashboard fraude
* **Attendu** :

  * Analyse massive

---

## 18. Système de vision pour agriculture intelligente

* **Type** : Segmentation d’image
* **Dataset** :
  [https://www.kaggle.com/datasets/kumaresanmanickavelu/leaf-disease-segmentation](https://www.kaggle.com/datasets/kumaresanmanickavelu/leaf-disease-segmentation)
* **Technos** : U-Net
* **App** : Mobile app (photo plante)
* **Attendu** :

  * Diagnostic visuel

---

## 19. Système de détection de churn en streaming

* **Type** : Classification + Streaming
* **Dataset** :
  [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Technos** : Kafka + Spark + ML
* **App** : Dashboard temps réel
* **Attendu** :

  * Score churn live

---

## 20. Plateforme d’analyse financière (crypto / stocks)

* **Type** : Time series + RL
* **Dataset** :
  [https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
* **Technos** : LSTM + RL
* **App** : Web dashboard trading
* **Attendu** :

  * Prédiction + stratégie

---


---

# 📦 SECTION : LIVRABLES ET MODALITÉS DE SOUMISSION

## 👥 Composition des équipes

* Les projets doivent être réalisés en **équipe de 2 à 3 étudiants maximum**
* Aucun groupe de 1 étudiant n’est autorisé (sauf exception validée)
* Chaque membre doit avoir un **rôle clairement défini** :

  * Data Engineer
  * ML Engineer
  * Backend / Frontend Developer
  * (ou autre répartition pertinente)

---

## 📄 Rapport final (OBLIGATOIRE)

Le rapport doit être soumis **avant la dernière séance** à l’adresse suivante :
📧 **tchaye59@gmail.com**

### 🧾 Format du rapport

* Format PDF
* Minimum : **5 pages**
* Maximum : **20 pages**
* Structure claire et professionnelle

---

## 🧱 Structure attendue du rapport

### 1. Introduction

* Contexte du projet
* Problématique métier
* Objectifs

---

### 2. Description des données

* Source des datasets (avec liens)
* Analyse exploratoire (EDA)
* Problèmes rencontrés (données manquantes, biais…)

---

### 3. Architecture du système

* Schéma global (OBLIGATOIRE)
* Pipeline data (batch / streaming)
* Outils utilisés (Spark, Hadoop, API, etc.)

---

### 4. Modélisation

* Choix des modèles (justification)
* Méthodologie
* Entraînement

---

### 5. Évaluation

* Métriques utilisées (F1, RMSE, etc.)
* Résultats obtenus
* Comparaison entre modèles

---

### 6. Déploiement

* API (FastAPI / Flask)
* Infrastructure (Docker, cloud ou local)
* Description de l’application (web/mobile)

---

### 7. Démonstration du système

* Captures d’écran de l’application
* Cas d’usage réel
* Résultats en situation

---

### 8. Limites et améliorations

* Points faibles
* Perspectives d’évolution

---

### 9. Répartition du travail (TRÈS IMPORTANT ⚠️)

Chaque groupe doit inclure une section détaillant :

* Nom de chaque membre
* Rôle de chaque membre
* Contributions précises (ex : modèle, API, frontend…)

👉 **Toute absence de répartition claire entraînera une pénalité**

---

## 💻 Code source (OBLIGATOIRE)

Les étudiants doivent fournir :

* Un **repository GitHub**
* Code propre et documenté
* README avec :

  * Instructions d’installation
  * Instructions d’exécution
  * Architecture du projet

---

## 🚀 Application finale (OBLIGATOIRE)

Chaque projet doit inclure une application fonctionnelle :

* Web (Streamlit / React / Flask) ✅
* OU Mobile (Flutter / Android) ✅
* OU Desktop ✅

👉 L’application doit permettre de :

* Tester le modèle
* Visualiser les résultats
* Interagir avec le système

---

## 🎤 Présentation finale

Chaque groupe devra :

* Présenter pendant **15 à 20 minutes**
* Faire une **démonstration live**
* Expliquer :

  * Architecture
  * Choix techniques
  * Résultats

---

## ⛔ Règles importantes

* ❌ Plagiat interdit (code ou rapport)
* ❌ Copie de projet Kaggle sans adaptation = 0
* ❌ Modèle seul sans application = REFUSÉ

---

## 📅 Deadline

* Soumission du rapport + code : **avant la dernière séance**
* Aucun retard accepté sans justification valable

---