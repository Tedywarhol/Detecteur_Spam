📩 Détecteur de Spam SMS avec Interface Graphique

Ce projet Python est une application de classification de SMS visant à détecter les spams à l’aide du modèle Naive Bayes.
Il intègre à la fois un prétraitement du texte, un entraînement de modèle supervisé et une interface utilisateur graphique (GUI) réalisée avec Tkinter.


---

🔍 Fonctionnalités principales

🧹 Nettoyage et vectorisation du texte (TF-IDF)

🧠 Entraînement automatique d’un modèle de classification Multinomial Naive Bayes

🧪 Évaluation des performances (précision, matrice de confusion, rapport de classification)

📂 Sauvegarde et chargement du modèle entraîné (pickle)

🖼 Interface graphique intuitive pour tester de nouveaux messages



---

📁 Structure du projet

spam_classifier/

│

├── spam.csv                    # Données d'entraînement (SMS taggés "spam" ou "ham")

├── spam_classifier.pkl         # (Généré) Modèle entraîné sauvegardé

├── main.py                     # Code principal avec classifieur + interface

└── README.md                   # Ce fichier


---

⚙ Étapes exécutées par le programmef

1. 📥 Chargement des données (spam.csv)

Le fichier CSV contient deux colonnes :

v1 : étiquette (ham ou spam)

v2 : message texte


Celles-ci sont renommées en label et text.


---

2. 🧹 Prétraitement des messages

Le texte est :

Converti en minuscules

Nettoyé de la ponctuation, chiffres et espaces superflus

Privé de stopwords anglais (via NLTK)

Vectorisé avec TF-IDF



---

3. 📊 Préparation des données

Les étiquettes sont transformées (ham → 0, spam → 1)

Le jeu de données est divisé en données d'entraînement et de test (80/20) avec préservation du ratio ham/spam



---

4. 🧠 Entraînement du modèle

Le modèle MultinomialNB est entraîné

Optionnel : optimisation automatique de l'hyperparamètre alpha via GridSearchCV



---

5. 📈 Évaluation du modèle

Le modèle est évalué avec :

Précision

Matrice de confusion

Rapport de classification (précision, rappel, f1-score)



---

6. 📂 Sauvegarde et chargement

Le modèle et le vectoriseur sont sauvegardés dans un fichier .pkl et peuvent être rechargés pour une utilisation ultérieure.


---

7. 🖼 Interface utilisateur (GUI)

L’utilisateur peut entrer un message dans une fenêtre Tkinter. En cliquant sur "Vérifier", le système indique si le message est :

✅ HAM (légitime)

🚨 SPAM



---

▶ Exécution

Prérequis

pip install pandas scikit-le
