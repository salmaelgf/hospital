# Système de Gestion Hospitalière avec Intelligence Artificielle

Ce projet est une plateforme complète de gestion hospitalière intégrant :

- Un Dashboard Flutter (Web & Mobile)
- Un Backend Spring Boot (API REST)
- Un Module d’Intelligence Artificielle en Python avec Flask
- Une Architecture basée sur les Git Submodules

Le système permet aux médecins de :
- Gérer les patients
- Consulter les dossiers médicaux
- Obtenir des prédictions d’adhérence thérapeutique basées sur l’IA

---

## Structure du Projet

```bash
hospital/
│
├── HospitalManagementDashboard/
│   ├── lib/
│   ├── web/
│   ├── pubspec.yaml
│   └── main.dart
│
├── adherence-backend/
│   ├── src/main/java/
│   ├── src/main/resources/
│   ├── pom.xml
│   └── mvnw
│
├── pfa5/
│   ├── app.py
│   ├── train_model.py
│   ├── predict_xgb.py
│   ├── *.pkl
│   └── *.csv
│
└── README.md
Prérequis
Avant d’exécuter le projet, assurez-vous d’avoir installé :

Flutter

Java JDK 17 ou supérieur

Maven

Python 3.9 ou supérieur

Git

1. Lancer le Module IA (Python / Flask)
bash
Copy code
cd pfa5
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
API accessible sur :

text
Copy code
http://127.0.0.1:5000
2. Lancer le Backend Spring Boot
bash
Copy code
cd adherence-backend
mvnw spring-boot:run
Ou :

bash
Copy code
mvn spring-boot:run
Backend accessible sur :

text
Copy code
http://localhost:8080
Endpoints de l’API
Méthode	Endpoint	Description
GET	/patients	Liste de tous les patients
GET	/patient/{id}	Détails d’un patient
GET	/predict/{id}	Prédiction d’adhérence au traitement

3. Lancer le Dashboard Flutter (Web)
bash
Copy code
cd HospitalManagementDashboard
flutter pub get
flutter run -d chrome
L’application s’ouvrira automatiquement dans le navigateur.

Fonctionnement Global du Système
text
Copy code
[Flutter] → [Spring Boot] → [Flask IA] → [Spring Boot] → [Flutter]
Le Dashboard Flutter envoie les requêtes

Le Backend Spring Boot appelle l’API IA

L’API IA retourne le score de prédiction

Le score est affiché dans le Dashboard

Technologies Utilisées
Frontend
Flutter

Dart

Web

Backend
Spring Boot

Java

API REST

Maven

Intelligence Artificielle
Python

Flask

Scikit-learn

XGBoost

Pickle (.pkl)

Sécurité & Communication
Communication via API REST

Séparation complète Frontend / Backend / IA

Services indépendants

Données traitées par identifiant patient uniquement

Cloner le Projet avec Submodules
bash
Copy code
git clone --recurse-submodules https://github.com/salmaelgf/hospital.git
Objectif du Projet
Digitaliser la gestion hospitalière

Exploiter l’intelligence artificielle dans la décision médicale

Réduire le risque de non-adhérence au traitement

Offrir une plateforme moderne, performante et évolutive

