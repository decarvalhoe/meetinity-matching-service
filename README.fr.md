# Service de Matching Meetinity

Ce repository contient le service de matching de la plateforme Meetinity, responsable de connecter les utilisateurs basé sur leurs profils professionnels et leurs intérêts.

## Vue d'ensemble

Le Service de Matching est développé avec **Python Flask** et fournit des algorithmes pour le matching d'utilisateurs, les suggestions de profils et les interactions basées sur le swipe. Il sert de composant central pour les recommandations de networking professionnel sur la plateforme Meetinity.

## Fonctionnalités

- **Algorithme de Matching Utilisateur** : Matching intelligent basé sur les profils professionnels, compétences et intérêts
- **Fonctionnalité Swipe** : Interface de swipe similaire à Tinder pour les interactions utilisateur (like/pass)
- **Suggestions de Profils** : Recommandations personnalisées avec scoring et raisonnement
- **Détection de Match** : Détection de match en temps réel quand un intérêt mutuel est établi
- **Scoring de Compatibilité** : Système de scoring avancé basé sur plusieurs facteurs

## Stack Technique

- **Flask** : Framework web Python léger
- **Python** : Algorithmes de matching centraux et logique métier

## État du Projet

- **Avancement** : 25%
- **Fonctionnalités terminées** : Structure API de base, points de données fictives, logique de swipe simple
- **Fonctionnalités en attente** : Intégration base de données réelle, algorithmes de matching avancés, recommandations machine learning, préférences utilisateur

## Implémentation Actuelle

Le service fournit actuellement des données fictives et des fonctionnalités de base :

- **Profils Fictifs** : Exemples de profils utilisateur avec informations professionnelles
- **Scoring de Base** : Scoring de compatibilité simple basé sur des critères prédéfinis
- **Logique Swipe** : Fonctionnalité like/pass de base avec détection de match

## Points d'accès API

- `GET /health` - Contrôle de santé du service
- `GET /matches/<user_id>` - Obtenir les matches pour un utilisateur spécifique
- `POST /swipe` - Enregistrer une action de swipe (like/pass)
- `GET /algorithm/suggest/<user_id>` - Obtenir des suggestions de profils personnalisées

## Pour Commencer

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Lancer le service :
   ```bash
   python src/main.py
   ```

Le service démarrera sur le port 5004 par défaut.

## Entraînement du Modèle de Préférences

Les swipes et les matches confirmés sont enregistrés dans la base SQLite. Pour
entraîner le modèle de préférence utilisé par la fonction
`predict_preference_score`, exécutez :

```bash
python scripts/train_preferences.py --refresh-settings
```

La commande collecte les derniers événements, réalise une séparation
train/test, puis stocke le modèle et ses métadonnées dans le dossier `models/`.
Le fichier `models/latest.json` référence automatiquement la version active et
permet à l'API de charger le bon artefact.

Planifiez l'entraînement régulièrement (par exemple toutes les nuits à 02h00)
via une tâche cron afin de garder les prédictions à jour :

```
0 2 * * * /usr/bin/python /chemin/vers/repo/scripts/train_preferences.py >> /var/log/meetinity/train.log 2>&1
```

Après chaque exécution, le service utilise immédiatement le nouveau modèle sans
redémarrage.

## Feuille de Route de Développement

### Phase 1 (Actuelle)
- Structure API de base avec données fictives
- Fonctionnalité swipe simple
- Surveillance de santé

### Phase 2 (Prochaine)
- Intégration base de données pour données utilisateur réelles
- Algorithmes de matching améliorés
- Gestion des préférences utilisateur

### Phase 3 (Future)
- Recommandations basées sur machine learning
- Métriques de compatibilité avancées
- Mises à jour de matching en temps réel

## Architecture

```
src/
├── main.py              # Point d'entrée de l'application
├── models/              # Modèles de données (à implémenter)
├── algorithms/          # Algorithmes de matching (à implémenter)
└── routes/              # Points d'accès API (à implémenter)
```

## Tests

```bash
pytest
flake8 src tests
```
