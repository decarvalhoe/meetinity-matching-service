# Meetinity Matching Service - Infrastructure Documentation

## Service Overview

The **Meetinity Matching Service** implements intelligent professional matching algorithms to connect users based on compatibility, shared interests, industry alignment, and networking preferences. This service uses machine learning models and real-time data processing to provide personalized matching recommendations.

## Architecture Position

```mermaid
graph TD
    A[Mobile App :3000] --> B[API Gateway :8080]
    C[Admin Portal :3001] --> B
    B --> D[User Service :8081]
    B --> E[Event Service :8083]
    B --> F[Matching Service :8082]
    
    D --> G[PostgreSQL :5432]
    E --> G
    F --> G
    
    D --> H[Redis :6379]
    F --> H
    E --> I[Kafka :9092]
    D --> I
    F --> I
    
    I --> J[Zookeeper :2181]
    
    F --> K[ML Models Storage]
    F --> L[Vector Database]
```

## Infrastructure Requirements

### System Dependencies
- **Python 3.11+** - Runtime environment
- **PostgreSQL 15** - Primary database
- **Redis** - Caching and real-time data
- **Apache Kafka 3.5** - Event streaming
- **Zookeeper 3.9** - Kafka coordination
- **NumPy/SciPy** - Scientific computing
- **scikit-learn** - Machine learning
- **Docker** - Containerization (optional)

### Resource Requirements
- **CPU**: 4 cores minimum, 8 cores recommended (ML workloads)
- **Memory**: 4GB minimum, 8GB recommended (ML models)
- **Storage**: 20GB minimum for models and feature data
- **GPU**: Optional, for advanced ML model training
- **Network**: 1Gbps for real-time matching

## Database Schema

### User Matches Table
```sql
CREATE TABLE user_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user1_id UUID NOT NULL, -- References users(id) from User Service
    user2_id UUID NOT NULL, -- References users(id) from User Service
    compatibility_score DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    match_reason JSONB, -- Detailed matching factors
    status VARCHAR(20) DEFAULT 'pending', -- pending, mutual, declined, expired
    algorithm_version VARCHAR(10) NOT NULL,
    matched_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '30 days'),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user1_id, user2_id),
    CHECK (user1_id != user2_id),
    CHECK (compatibility_score >= 0.0 AND compatibility_score <= 1.0)
);

-- Indexes for performance
CREATE INDEX idx_matches_user1 ON user_matches(user1_id);
CREATE INDEX idx_matches_user2 ON user_matches(user2_id);
CREATE INDEX idx_matches_status ON user_matches(status);
CREATE INDEX idx_matches_score ON user_matches(compatibility_score DESC);
CREATE INDEX idx_matches_expires ON user_matches(expires_at);
```

### Swipe History Table
```sql
CREATE TABLE swipe_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    swiper_id UUID NOT NULL, -- User who performed the swipe
    swiped_id UUID NOT NULL, -- User who was swiped on
    action VARCHAR(10) NOT NULL, -- 'like', 'pass', 'super_like'
    match_id UUID REFERENCES user_matches(id), -- If this created a match
    context JSONB, -- Additional context (location, event, etc.)
    swiped_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(swiper_id, swiped_id)
);

-- Indexes
CREATE INDEX idx_swipes_swiper ON swipe_history(swiper_id);
CREATE INDEX idx_swipes_swiped ON swipe_history(swiped_id);
CREATE INDEX idx_swipes_action ON swipe_history(action);
CREATE INDEX idx_swipes_date ON swipe_history(swiped_at DESC);
```

### User Preferences Table
```sql
CREATE TABLE matching_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL, -- References users(id) from User Service
    age_min INTEGER DEFAULT 18,
    age_max INTEGER DEFAULT 65,
    distance_km INTEGER DEFAULT 50,
    industries TEXT[],
    job_levels TEXT[], -- entry, mid, senior, executive, founder
    company_sizes TEXT[], -- startup, small, medium, large, enterprise
    interests TEXT[],
    networking_goals TEXT[], -- career_growth, mentorship, collaboration, investment
    availability JSONB, -- When user is available for networking
    preferred_meeting_types TEXT[], -- coffee, lunch, virtual, events
    languages TEXT[],
    exclude_colleagues BOOLEAN DEFAULT TRUE,
    exclude_competitors BOOLEAN DEFAULT FALSE,
    show_distance BOOLEAN DEFAULT TRUE,
    algorithm_weights JSONB, -- Custom weights for matching factors
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_preferences_user ON matching_preferences(user_id);
CREATE INDEX idx_preferences_distance ON matching_preferences(distance_km);
CREATE INDEX idx_preferences_industries ON matching_preferences USING GIN(industries);
CREATE INDEX idx_preferences_interests ON matching_preferences USING GIN(interests);
```

### User Features Table (ML Features)
```sql
CREATE TABLE user_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL,
    feature_vector FLOAT8[], -- High-dimensional feature vector
    profile_completeness DECIMAL(3, 2), -- 0.00 to 1.00
    activity_score DECIMAL(5, 4), -- User activity level
    network_size INTEGER DEFAULT 0,
    response_rate DECIMAL(3, 2), -- Historical response rate
    industry_embedding FLOAT8[],
    skill_embedding FLOAT8[],
    location_embedding FLOAT8[],
    computed_at TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(10) NOT NULL
);

-- Indexes
CREATE INDEX idx_features_user ON user_features(user_id);
CREATE INDEX idx_features_completeness ON user_features(profile_completeness DESC);
CREATE INDEX idx_features_activity ON user_features(activity_score DESC);
CREATE INDEX idx_features_computed ON user_features(computed_at DESC);
```

### Matching Analytics Table
```sql
CREATE TABLE matching_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- match_shown, match_liked, match_passed, conversation_started
    match_id UUID REFERENCES user_matches(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for analytics
CREATE INDEX idx_analytics_user ON matching_analytics(user_id);
CREATE INDEX idx_analytics_type ON matching_analytics(event_type);
CREATE INDEX idx_analytics_date ON matching_analytics(created_at DESC);
CREATE INDEX idx_analytics_match ON matching_analytics(match_id);
```

## Service Configuration

### Environment Variables
```env
# Service Configuration
APP_PORT=8082
LOG_LEVEL=INFO
FLASK_ENV=production

# Database Configuration
DATABASE_URL=postgresql://meetinity:meetinity@localhost:5432/meetinity

# Redis Configuration
REDIS_URL=redis://localhost:6379/1

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_MATCHING_TOPIC=matching.events
KAFKA_CONSUMER_GROUP=matching-service

# External Service URLs
USER_SERVICE_URL=http://localhost:8081
EVENT_SERVICE_URL=http://localhost:8083
API_GATEWAY_URL=http://localhost:8080

# Machine Learning Configuration
ML_MODEL_PATH=/app/models
ML_MODEL_VERSION=v1.2.0
FEATURE_DIMENSION=512
SIMILARITY_THRESHOLD=0.7
MAX_MATCHES_PER_USER=50

# Matching Algorithm Configuration
ALGORITHM_WEIGHTS_PROFILE=0.3
ALGORITHM_WEIGHTS_INTERESTS=0.25
ALGORITHM_WEIGHTS_LOCATION=0.2
ALGORITHM_WEIGHTS_INDUSTRY=0.15
ALGORITHM_WEIGHTS_ACTIVITY=0.1

# Performance Configuration
BATCH_SIZE=100
MAX_CONCURRENT_MATCHES=1000
CACHE_TTL=1800  # 30 minutes

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:8080
```

### Docker Configuration
```dockerfile
# Multi-stage build for ML dependencies
FROM python:3.11-slim as builder
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime
RUN apt-get update && apt-get install -y \
    curl \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r meetinity && useradd -r -g meetinity meetinity
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --chown=meetinity:meetinity . .
RUN mkdir -p /app/models && chown meetinity:meetinity /app/models

USER meetinity
EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CMD ["gunicorn", "--bind", "0.0.0.0:8082", "--workers", "2", "--timeout", "120", "src.main:create_app()"]
```

## Machine Learning Architecture

### Feature Engineering Pipeline
```python
class UserFeatureExtractor:
    def extract_features(self, user_profile):
        features = {
            # Profile completeness features
            'profile_completeness': self.calculate_completeness(user_profile),
            'has_photo': bool(user_profile.get('profile_picture_url')),
            'bio_length': len(user_profile.get('bio', '')),
            
            # Professional features
            'industry_encoded': self.encode_industry(user_profile.get('industry')),
            'job_level': self.encode_job_level(user_profile.get('job_title')),
            'company_size': self.encode_company_size(user_profile.get('company')),
            
            # Geographic features
            'location_lat': user_profile.get('latitude', 0),
            'location_lng': user_profile.get('longitude', 0),
            'location_density': self.get_location_density(user_profile.get('location')),
            
            # Activity features
            'days_since_signup': self.days_since_signup(user_profile.get('created_at')),
            'last_activity_days': self.days_since_activity(user_profile.get('last_login')),
            'event_attendance': self.get_event_count(user_profile.get('id')),
            
            # Network features
            'connection_count': self.get_connection_count(user_profile.get('id')),
            'mutual_connections': self.get_mutual_connections(user_profile.get('id')),
            
            # Behavioral features
            'swipe_ratio': self.get_swipe_ratio(user_profile.get('id')),
            'response_rate': self.get_response_rate(user_profile.get('id')),
            'conversation_starter_rate': self.get_conversation_rate(user_profile.get('id'))
        }
        
        return self.normalize_features(features)
```

### Matching Algorithm
```python
class CompatibilityScorer:
    def __init__(self):
        self.weights = {
            'profile_similarity': 0.3,
            'interest_overlap': 0.25,
            'location_proximity': 0.2,
            'industry_alignment': 0.15,
            'activity_compatibility': 0.1
        }
    
    def calculate_compatibility(self, user1_features, user2_features, preferences):
        scores = {}
        
        # Profile similarity using cosine similarity
        scores['profile_similarity'] = cosine_similarity(
            user1_features['profile_vector'],
            user2_features['profile_vector']
        )
        
        # Interest overlap using Jaccard similarity
        scores['interest_overlap'] = self.jaccard_similarity(
            user1_features['interests'],
            user2_features['interests']
        )
        
        # Location proximity (inverse distance)
        scores['location_proximity'] = self.calculate_location_score(
            user1_features['location'],
            user2_features['location'],
            preferences['distance_km']
        )
        
        # Industry alignment
        scores['industry_alignment'] = self.calculate_industry_score(
            user1_features['industry'],
            user2_features['industry']
        )
        
        # Activity compatibility
        scores['activity_compatibility'] = self.calculate_activity_score(
            user1_features['activity_level'],
            user2_features['activity_level']
        )
        
        # Weighted final score
        final_score = sum(
            scores[factor] * self.weights[factor]
            for factor in scores
        )
        
        return final_score, scores
```

### Model Training Pipeline
```python
class MatchingModelTrainer:
    def train_compatibility_model(self):
        # Load training data
        positive_pairs = self.load_successful_matches()
        negative_pairs = self.load_unsuccessful_matches()
        
        # Feature extraction
        X_positive = self.extract_pair_features(positive_pairs)
        X_negative = self.extract_pair_features(negative_pairs)
        
        # Create training dataset
        X = np.vstack([X_positive, X_negative])
        y = np.hstack([np.ones(len(X_positive)), np.zeros(len(X_negative))])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate model
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Model accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        # Save model
        joblib.dump(model, f"{self.model_path}/compatibility_model_v{self.version}.pkl")
        
        return model
```

## Inter-Service Communication

### Kafka Events Published
```python
# Match created
{
    "event_type": "match.created",
    "match_id": "uuid",
    "user1_id": "user-uuid-1",
    "user2_id": "user-uuid-2",
    "compatibility_score": 0.8542,
    "match_reason": {
        "primary_factors": ["industry_alignment", "interest_overlap"],
        "shared_interests": ["AI", "Startups"],
        "industry_match": "Technology",
        "location_distance_km": 5.2
    },
    "timestamp": "2025-09-28T16:30:00Z"
}

# Swipe performed
{
    "event_type": "swipe.performed",
    "swiper_id": "user-uuid",
    "swiped_id": "user-uuid-2",
    "action": "like",
    "context": {
        "source": "discovery",
        "compatibility_score": 0.7823,
        "position_in_stack": 3
    },
    "timestamp": "2025-09-28T16:30:00Z"
}

# Match status updated
{
    "event_type": "match.status_updated",
    "match_id": "uuid",
    "old_status": "pending",
    "new_status": "mutual",
    "updated_by": "user-uuid",
    "timestamp": "2025-09-28T16:30:00Z"
}
```

### Kafka Events Consumed
```python
# User profile updated
{
    "event_type": "user.updated",
    "user_id": "uuid",
    "updated_fields": ["industry", "interests", "location"],
    "old_values": {...},
    "new_values": {...},
    "timestamp": "2025-09-28T16:30:00Z"
}

# Event attendance
{
    "event_type": "event.registration.created",
    "event_id": "uuid",
    "user_id": "user-uuid",
    "event_category": "Networking",
    "event_tags": ["tech", "startup"],
    "timestamp": "2025-09-28T16:30:00Z"
}

# User deleted
{
    "event_type": "user.deleted",
    "user_id": "uuid",
    "timestamp": "2025-09-28T16:30:00Z"
}
```

### API Integration Points

#### User Service Integration
```python
# Get user profile with preferences
GET /users/{user_id}/profile?include=preferences,activity
Response: {
    "id": "uuid",
    "name": "John Doe",
    "industry": "Technology",
    "interests": ["AI", "Blockchain"],
    "location": {"lat": 37.7749, "lng": -122.4194},
    "preferences": {...},
    "activity": {...}
}

# Batch user profiles for matching
POST /users/batch?include=preferences
Body: {"user_ids": ["uuid1", "uuid2", "uuid3"]}
Response: [{"id": "uuid1", ...}, ...]

# Validate user compatibility
POST /users/compatibility-check
Body: {"user1_id": "uuid1", "user2_id": "uuid2"}
Response: {"compatible": true, "restrictions": []}
```

#### Event Service Integration
```python
# Get user's event history for matching context
GET /events/users/{user_id}/history
Response: {
    "attended_events": [...],
    "created_events": [...],
    "preferred_categories": ["Networking", "Workshop"],
    "activity_score": 0.75
}

# Get event attendees for potential matches
GET /events/{event_id}/attendees?include=profiles
Response: {
    "attendees": [{"user_id": "uuid", "profile": {...}}, ...]
}
```

## Deployment Architecture

### Local Development
```bash
# Start dependencies
docker-compose up -d postgres redis kafka zookeeper

# Install ML dependencies
pip install numpy scipy scikit-learn joblib

# Start service
python src/main.py
```

### Docker Compose Integration
```yaml
version: "3.9"
services:
  matching-service:
    build: ../meetinity-matching-service
    ports: ["8082:8082"]
    environment:
      - DATABASE_URL=postgresql://meetinity:meetinity@postgres:5432/meetinity
      - REDIS_URL=redis://redis:6379/1
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - USER_SERVICE_URL=http://user-service:8081
      - EVENT_SERVICE_URL=http://event-service:8083
    volumes:
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
      - kafka
      - user-service
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8082/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: matching-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: matching-service
  template:
    metadata:
      labels:
        app: matching-service
    spec:
      containers:
      - name: matching-service
        image: meetinity/matching-service:latest
        ports:
        - containerPort: 8082
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          value: "redis://redis:6379/1"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8082
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8082
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: ml-models-pvc
```

## Performance Optimization

### Caching Strategy
```python
# Redis caching for expensive computations
CACHE_KEYS = {
    "user_features": "features:{user_id}",
    "user_matches": "matches:{user_id}",
    "compatibility_matrix": "compat:{user1_id}:{user2_id}",
    "popular_profiles": "popular:{location}:{industry}",
    "ml_predictions": "ml:{model_version}:{feature_hash}"
}

CACHE_TIMEOUTS = {
    "user_features": 3600,          # 1 hour
    "user_matches": 1800,           # 30 minutes
    "compatibility_matrix": 7200,   # 2 hours
    "popular_profiles": 3600,       # 1 hour
    "ml_predictions": 86400         # 24 hours
}
```

### Database Optimization
```sql
-- Partitioning for large matching tables
CREATE TABLE swipe_history_2025 PARTITION OF swipe_history
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Materialized view for matching statistics
CREATE MATERIALIZED VIEW matching_stats AS
SELECT 
    DATE_TRUNC('day', matched_at) as match_date,
    COUNT(*) as total_matches,
    AVG(compatibility_score) as avg_score,
    COUNT(*) FILTER (WHERE status = 'mutual') as mutual_matches
FROM user_matches 
GROUP BY DATE_TRUNC('day', matched_at);

-- Index for fast similarity searches
CREATE INDEX idx_features_vector ON user_features USING ivfflat (feature_vector vector_cosine_ops);
```

### Batch Processing
```python
class BatchMatchingProcessor:
    def process_daily_matches(self):
        # Get active users
        active_users = self.get_active_users()
        
        # Process in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(active_users), batch_size):
            batch = active_users[i:i + batch_size]
            self.process_user_batch(batch)
    
    def process_user_batch(self, users):
        # Extract features for all users in batch
        features = self.extract_batch_features(users)
        
        # Calculate compatibility matrix
        compatibility_matrix = self.calculate_batch_compatibility(features)
        
        # Generate matches
        matches = self.generate_batch_matches(compatibility_matrix)
        
        # Store matches in database
        self.store_batch_matches(matches)
```

## Monitoring and Observability

### Health Check Endpoint
```python
GET /health
Response:
{
    "status": "ok",
    "service": "matching-service",
    "version": "1.0.0",
    "dependencies": {
        "database": "up",
        "redis": "up",
        "kafka": "up",
        "user_service": "up",
        "event_service": "up"
    },
    "ml_models": {
        "compatibility_model": {
            "version": "v1.2.0",
            "loaded": true,
            "accuracy": 0.847
        },
        "feature_extractor": {
            "version": "v1.1.0",
            "loaded": true
        }
    },
    "metrics": {
        "total_matches": 15000,
        "active_matches": 3500,
        "daily_swipes": 1200,
        "match_success_rate": 0.23
    }
}
```

### ML Model Monitoring
```python
class ModelMonitor:
    def monitor_model_performance(self):
        # Track prediction accuracy
        recent_predictions = self.get_recent_predictions()
        actual_outcomes = self.get_actual_outcomes()
        
        accuracy = self.calculate_accuracy(recent_predictions, actual_outcomes)
        precision = self.calculate_precision(recent_predictions, actual_outcomes)
        recall = self.calculate_recall(recent_predictions, actual_outcomes)
        
        # Alert if performance degrades
        if accuracy < self.accuracy_threshold:
            self.send_alert("Model accuracy below threshold", accuracy)
        
        # Log metrics
        self.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "timestamp": datetime.utcnow()
        })
```

## Security Considerations

### Data Privacy
```python
# Anonymize sensitive data in ML features
class PrivacyPreserver:
    def anonymize_features(self, user_features):
        # Remove direct identifiers
        anonymized = user_features.copy()
        anonymized.pop('email', None)
        anonymized.pop('phone', None)
        anonymized.pop('full_name', None)
        
        # Hash sensitive fields
        if 'company' in anonymized:
            anonymized['company_hash'] = self.hash_field(anonymized.pop('company'))
        
        # Add noise to location data
        if 'location' in anonymized:
            anonymized['location'] = self.add_location_noise(anonymized['location'])
        
        return anonymized
```

### Rate Limiting
```python
RATE_LIMITS = {
    "/matches/discover": "50/hour",      # Match discovery
    "/matches/swipe": "200/hour",        # Swipe actions
    "/matches/preferences": "10/hour",   # Preference updates
    "/matches/analytics": "100/hour"     # Analytics queries
}
```

## Related Documentation

- **API Gateway**: [meetinity-api-gateway](https://github.com/decarvalhoe/meetinity-api-gateway)
- **User Service**: [meetinity-user-service](https://github.com/decarvalhoe/meetinity-user-service)
- **Event Service**: [meetinity-event-service](https://github.com/decarvalhoe/meetinity-event-service)
- **Mobile App**: [meetinity-mobile-app](https://github.com/decarvalhoe/meetinity-mobile-app)
- **Admin Portal**: [meetinity-admin-portal](https://github.com/decarvalhoe/meetinity-admin-portal)
- **Main Repository**: [meetinity](https://github.com/decarvalhoe/meetinity)

---

**Status**: 25% Complete - Basic algorithms implemented  
**Last Updated**: September 2025  
**Maintainer**: decarvalhoe
