import pandas as pd
import re
import nltk
import pickle
import tkinter as tk
from tkinter import messagebox
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Téléchargement des stopwords
nltk.download('stopwords')


class SpamClassifier:
    """Classe principale pour la détection de spam"""

    def __init__(self):
        """Initialise le classifieur avec les composants principaux"""
        self.model = None
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words('english'))
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, file_path):
        """
        Charge les données à partir d'un fichier CSV
        Args:
            file_path (str): Chemin vers le fichier CSV
        Returns:
            pd.DataFrame: DataFrame contenant les colonnes 'label' et 'text'
        """
        try:
            # Chargement des données avec encodage latin-1
            df = pd.read_csv(file_path, encoding='latin-1')[['v1', 'v2']]
            df.columns = ['label', 'text']  # Renommage des colonnes

            # Affichage d'informations sur les données
            print("\nAperçu des données :")
            print(df.head())

            print("\nDistribution des classes :")
            print(df['label'].value_counts())

            return df
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            raise

    def clean_text(self, text):
        """
        Nettoie le texte en appliquant plusieurs transformations
        Args:
            text (str): Texte à nettoyer
        Returns:
            str: Texte nettoyé
        """
        try:
            # Conversion en minuscules
            text = text.lower()

            # Suppression de la ponctuation et des caractères spéciaux
            text = re.sub(r'\W', ' ', text)

            # Suppression des chiffres
            text = re.sub(r'\d+', '', text)

            # Normalisation des espaces
            text = re.sub(r'\s+', ' ', text).strip()

            # Suppression des stopwords
            words = text.split()
            words = [word for word in words if word not in self.stop_words]

            return ' '.join(words)
        except Exception as e:
            print(f"Erreur lors du nettoyage du texte: {e}")
            return text

    def prepare_data(self, df):
        """
        Prépare les données pour l'entraînement
        Args:
            df (pd.DataFrame): DataFrame contenant les données brutes
        Returns:
            tuple: (X, y) matrices de features et labels
        """
        try:
            # Nettoyage du texte
            df['clean_text'] = df['text'].apply(self.clean_text)

            print("\nExemple de texte nettoyé :")
            print(df[['text', 'clean_text']].head())

            # Encodage des labels (ham:0, spam:1)
            df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

            # Vectorisation TF-IDF
            X = self.vectorizer.fit_transform(df['clean_text'])
            y = df['label_num']

            print("\nForme de la matrice TF-IDF :", X.shape)

            return X, y
        except Exception as e:
            print(f"Erreur lors de la préparation des données: {e}")
            raise

    def split_data(self, X, y):
        """
        Divise les données en ensembles d'entraînement et de test
        Args:
            X: Matrice de features
            y: Vecteur de labels
        """
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y  # Conservation de la distribution des classes
            )

            print("\nRépartition des données :")
            print(f"- Ensemble d'entraînement: {self.X_train.shape[0]} exemples")
            print(f"- Ensemble de test: {self.X_test.shape[0]} exemples")
        except Exception as e:
            print(f"Erreur lors de la division des données: {e}")
            raise

    def train_model(self, optimize_hyperparams=True):
        """
        Entraîne le modèle avec possibilité d'optimisation
        Args:
            optimize_hyperparams (bool): Active l'optimisation des hyperparamètres
        Returns:
            float: Score de précision
        """
        try:
            if optimize_hyperparams:
                # Optimisation avec validation croisée
                parameters = {'alpha': [0.1, 0.5, 1.0, 1.5]}
                self.model = GridSearchCV(
                    MultinomialNB(),
                    parameters,
                    cv=5,  # 5-fold cross-validation
                    scoring='accuracy'
                )
                self.model.fit(self.X_train, self.y_train)

                print("\n🔎 Meilleurs paramètres trouvés :")
                print(self.model.best_params_)
            else:
                # Entraînement standard
                self.model = MultinomialNB()
                self.model.fit(self.X_train, self.y_train)

            # Évaluation
            return self.evaluate_model()
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle: {e}")
            raise

    def evaluate_model(self):
        """Évalue les performances du modèle"""
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)

            print("\n📊 Résultats de l'évaluation :")
            print(f"Précision globale: {accuracy:.4f}")

            print("\n📝 Rapport de classification :")
            print(classification_report(self.y_test, y_pred))

            print("\n📉 Matrice de confusion :")
            print(confusion_matrix(self.y_test, y_pred))

            return accuracy
        except Exception as e:
            print(f"Erreur lors de l'évaluation du modèle: {e}")
            raise

    def save_model(self, file_path):
        """
        Sauvegarde le modèle et le vectoriseur
        Args:
            file_path (str): Chemin du fichier de sauvegarde
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer
                }, f)
            print(f"\n💾 Modèle sauvegardé dans {file_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            raise

    def load_model(self, file_path):
        """
        Charge un modèle pré-entraîné
        Args:
            file_path (str): Chemin du fichier à charger
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
            print(f"\n🔍 Modèle chargé depuis {file_path}")
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            raise

    def predict(self, message):
        """
        Prédit si un message est spam ou ham
        Args:
            message (str): Message à classifier
        Returns:
            str: 'spam' ou 'ham'
        """
        try:
            if not self.model or not self.vectorizer:
                raise ValueError("Modèle non initialisé. Veuillez d'abord entraîner ou charger un modèle.")

            # Prétraitement du texte
            cleaned_text = self.clean_text(message)

            # Vectorisation
            vectorized_text = self.vectorizer.transform([cleaned_text])

            # Prédiction
            prediction = self.model.predict(vectorized_text)[0]

            return 'spam' if prediction == 1 else 'ham'
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            raise


class SpamClassifierUI:
    """Interface graphique pour le classifieur de spam"""

    def __init__(self, classifier):
        """
        Initialise l'interface
        Args:
            classifier (SpamClassifier): Instance du classifieur
        """
        self.classifier = classifier

        # Configuration de la fenêtre principale
        self.window = tk.Tk()
        self.window.title("Détecteur de Spam")
        self.window.geometry("500x300")

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        """Crée et positionne les éléments de l'interface"""
        # Titre
        tk.Label(
            self.window,
            text="Vérificateur de Spam SMS",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Zone de texte
        tk.Label(self.window, text="Entrez votre message :").pack()
        self.text_entry = tk.Text(self.window, height=8, width=50)
        self.text_entry.pack(padx=10, pady=5)

        # Bouton de vérification
        tk.Button(
            self.window,
            text="Vérifier",
            command=self.check_spam,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10)
        ).pack(pady=10)

        # Label de résultat
        self.result_label = tk.Label(
            self.window,
            text="",
            font=("Arial", 12, "bold")
        )
        self.result_label.pack()

    def check_spam(self):
        """Analyse le message et affiche le résultat"""
        message = self.text_entry.get("1.0", tk.END).strip()

        if not message:
            messagebox.showwarning("Attention", "Veuillez entrer un message à analyser")
            return

        try:
            result = self.classifier.predict(message)

            # Affichage coloré du résultat
            if result == "spam":
                self.result_label.config(
                    text="Résultat: SPAM 🚨",
                    fg="red"
                )
            else:
                self.result_label.config(
                    text="Résultat: HAM (legitime) ✅",
                    fg="green"
                )
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue : {str(e)}")

    def run(self):
        """Lance l'application"""
        self.window.mainloop()


if __name__ == "__main__":
    # Initialisation du classifieur
    classifier = SpamClassifier()

    # 1. Charger les données
    df = classifier.load_data("spam.csv")

    # 2. Préparer les données
    X, y = classifier.prepare_data(df)

    # 3. Diviser les données
    classifier.split_data(X, y)

    # 4. Entraîner le modèle (avec optimisation)
    classifier.train_model(optimize_hyperparams=True)

    # 5. Sauvegarder le modèle
    classifier.save_model("spam_classifier.pkl")

    # 7. Lancer l'interface
    app = SpamClassifierUI(classifier)
    app.run()
