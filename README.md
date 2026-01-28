# Détecter des faux billets avec Python  💶🤖

Ce projet a pour objectif de développer un système automatisé de **détection de faux billets en euros** à partir de leurs **dimensions géométriques**, en utilisant des algorithmes de machine learning supervisé.

## 1. Contexte 🛡️

L’ONCFM (Organisation Nationale de Lutte contre le Faux-Monnayage) fait face à une augmentation du faux-monnayage.  
La détection manuelle est coûteuse, lente et sujette à l’erreur humaine.  
L’enjeu est de mettre en place un **algorithme fiable, interprétable et facilement déployable** pour aider les équipes de contrôle.

## 2. Objectif du projet 🎯

- Développer un **modèle de classification** capable de prédire si un billet est **authentique ou contrefait**.
- Utiliser **uniquement 6 mesures géométriques** (longueur, hauteurs, marges, diagonale).
- Comparer plusieurs modèles puis **retenir le meilleur compromis** entre performance et interprétabilité.
- Proposer un **script de prédiction** réutilisable en production.

## 3. Données 📊

- 1500 billets  
  - 1000 vrais ✅  
  - 500 faux ❌  
- 6 variables géométriques :
  - `length`
  - `height_left`
  - `height_right`
  - `margin_up`
  - `margin_low`
  - `diagonal`

Les données sont fournies dans le fichier :

- billets.csv

## 4. Méthodologie 🧠

Étapes principales :

1. **Préparation et exploration**
   - Nettoyage des données 🧹
   - Statistiques descriptives
   - Visualisations (distributions, boxplots, heatmap de corrélations)

2. **Modélisation supervisée**
   - Régression logistique
   - KNN
   - Random Forest

3. **Analyse non supervisée**
   - K-means (clustering en 2 groupes)

4. **Évaluation**
   - Accuracy
   - Précision, rappel, F1-score
   - Matrices de confusion

## 5. Résultats principaux 🚀

Les quatre modèles testés présentent des performances élevées (≈ 98,3 % à 99 % d’accuracy).

Le **modèle retenu** est :

> 🎯 **Régression logistique**  
> - Accuracy ≈ 99 %  
> - Seuls 2 faux billets prédits à tort comme vrais sur 300 billets de test  
> - Modèle simple, rapide, interprétable et robuste 💡  

KNN et Random Forest obtiennent également de très bons résultats mais avec légèrement plus d’erreurs sur les faux billets.  
K-means est utilisé comme **outil exploratoire** pour montrer l’existence de deux groupes naturels (vrais / faux), mais n’est pas retenu comme modèle de prédiction final.

## 6. Structure du dépôt 📁
6. Structure du dépôt 📁
bash
.
├── data/
│   ├── billets.csv
│   └── predictions_billets.csv
├── notebooks/
│   ├── P12_Detectez-des-faux-billets.ipynb
│   └── Script-P12.ipynb
├── src/
│   └── modele_billets_lr.py
├── models/
│   └── model_logistic_regression.joblib   # si tu ajoutes le modèle sauvegardé
├── reports/
│   ├── P12_Detecter-des-faux-billets-avec-Python.pdf
│   └── Cahier-des-charges-detection-faux-billets_P12_DAS.pdf
├── extras/
│   └── P12-oral.docx
├── profile/
│   └── profil_Machine-learning_les-billets.html
└── README.md

## 📎 Ressources complémentaires

- Notes de préparation à la soutenance : `extras/P12-oral.docx`
- Profil interactif machine learning : `profile/profil_Machine-learning_les-billets.html`
