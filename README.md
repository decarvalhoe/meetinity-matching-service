# Meetinity Matching Service

This repository contains the matching service for the Meetinity platform, responsible for connecting users based on their professional profiles and interests.

## Overview

The Matching Service is built with **Python Flask** and provides algorithms for user matching, profile suggestions, and swipe-based interactions. It serves as the core component for professional networking recommendations on the Meetinity platform.

## Features

- **User Matching Algorithm**: Intelligent matching based on professional profiles, skills, and interests
- **Swipe Functionality**: Tinder-like swipe interface for user interactions (like/pass)
- **Profile Suggestions**: Personalized recommendations with scoring and reasoning
- **Match Detection**: Real-time match detection when mutual interest is established
- **Compatibility Scoring**: Advanced scoring system based on multiple factors

## Tech Stack

- **Flask**: Lightweight Python web framework
- **Python**: Core matching algorithms and business logic

## Project Status

- **Progress**: 25%
- **Completed Features**: Basic API structure, mock data endpoints, simple swipe logic
- **Pending Features**: Real database integration, advanced matching algorithms, machine learning recommendations, user preferences

## Current Implementation

The service currently provides mock data and basic functionality:

- **Mock Profiles**: Sample user profiles with professional information
- **Basic Scoring**: Simple compatibility scoring based on predefined criteria
- **Swipe Logic**: Basic like/pass functionality with match detection

## API Endpoints

- `GET /health` - Service health check
- `GET /matches/<user_id>` - Get matches for a specific user
- `POST /swipe` - Record a swipe action (like/pass)
- `GET /algorithm/suggest/<user_id>` - Get personalized profile suggestions

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the service:
   ```bash
   python src/main.py
   ```

The service will start on port 5004 by default.

## Preference Model Training

Swipe events and confirmed matches are recorded in the embedded SQLite database.
You can export them and train the machine learning model that powers the
`predict_preference_score` inference helper by running:

```bash
python scripts/train_preferences.py --refresh-settings
```

This command collects the most recent swipe events, performs a train/test split,
and stores the resulting model (together with metadata and metrics) inside the
`models/` directory. The latest model is automatically referenced through a
`models/latest.json` file so that the API can use it for inference.

To keep the predictions up to date, schedule the script to run periodically.
For instance, on a Unix-like system you can add the following cron entry to
retrain the model every night at 02:00:

```
0 2 * * * /usr/bin/python /path/to/repo/scripts/train_preferences.py >> /var/log/meetinity/train.log 2>&1
```

After each training run the service immediately picks up the new model without
requiring a restart.

## Development Roadmap

### Phase 1 (Current)
- Basic API structure with mock data
- Simple swipe functionality
- Health monitoring

### Phase 2 (Next)
- Database integration for real user data
- Enhanced matching algorithms
- User preference management

### Phase 3 (Future)
- Machine learning-based recommendations
- Advanced compatibility metrics
- Real-time matching updates

## Architecture

```
src/
├── main.py              # Application entry point
├── models/              # Data models (to be implemented)
├── algorithms/          # Matching algorithms (to be implemented)
└── routes/              # API endpoints (to be implemented)
```

## Testing

```bash
pytest
flake8 src tests
```
