# 🍕 Chatbot Pizza

## 🧐 Contexte
En tant que développeur Intelligence Artificielle / Ingénieur IA Full-Stack, vous êtes missionné pour un projet de 1 journée visant à développer un assistant IA pour une pizzeria. Problématique : La pizzeria "Bella Napoli" fait face à des appels constants de clients souhaitant connaître les ingrédients des pizzas (allergies, préférences alimentaires). Le personnel est surchargé par ces demandes répétitives, ce qui impacte la qualité du service et fait perdre des commandes. 

## 🔍️ Fonctionnalités et spécificités
-  Application fonctionnelle permettant aux clients de poser des questions en langage naturel sur les ingrédients des pizzas et recevant des réponses précises extraites de la documentation fournie.
- Utilisation de l'architecture RAG (Retrieval-Augmented Generation) avec LangChain
- interface visuelle avec Gradio
- Base documentaire concçue avec des documents PDF fournis

## 🏗️ Architecture
```
chatbot_pizza/
        |--.gitignore
        |--app.py               # application gradio
        |--rag.py               # création de la base documentaire
        |--README.md
        |--requirements.txt     # dépendances
```


## 🚀 Lancement du projet

#### 1. 👥 Cloner le projet
```bash
git clone https://github.com/Nickaos1203/chatbot_pizza.git
cd chatbot_pizza
```
#### 2. ✅ Création et activation de l'environnement virtuel
```bash
python -m venv .venv
source .venv/Scripts/activate
```
Remarque : assurez-vous d'être sur Windows
#### 3. ✅ Installations des dépendances
```bash
pip install -r requirements.txt
```
#### 4. ✅ Création de la base documentation pour le RAG
```bash
python rag.py
```
Un dossier 'chroma_db' doit apparaitre à la racine du projet
#### 5. ✅ Lancement de l'application Gradio
```bash
python app.py
```
