# 🤝 Meetinity Matching Service

## ⚠️ **REPOSITORY ARCHIVED - MOVED TO MONOREPO**

**This repository has been archived and is now read-only.**

### 📍 **New Location**
All development has moved to the **Meetinity monorepo**:

**🔗 https://github.com/decarvalhoe/meetinity**

The Matching Service is now located at:
```
meetinity/services/matching-service/
```

### 🔄 **Migration Complete**
- ✅ **All code** migrated with complete history
- ✅ **ML algorithms** and scoring systems
- ✅ **Infrastructure documentation** and deployment guides
- ✅ **API integration** documentation
- ✅ **CI/CD pipeline** integrated with unified deployment

### 🛠️ **For Developers**

#### **Clone the monorepo:**
```bash
git clone https://github.com/decarvalhoe/meetinity.git
cd meetinity/services/matching-service
```

#### **Development workflow:**
```bash
# Start all services including ML dependencies
docker compose -f docker-compose.dev.yml up

# Matching Service specific development
cd services/matching-service
python scripts/train_preferences.py  # Train ML models
pytest                               # Run tests
```

### 📚 **Documentation**
- **Service Documentation**: `meetinity/services/matching-service/README.md`
- **Infrastructure Guide**: `meetinity/services/matching-service/INFRASTRUCTURE.md`
- **API Documentation**: `meetinity/services/matching-service/docs/`
- **ML Models**: `meetinity/services/matching-service/models/`

### 🤖 **Machine Learning Features**
Now available in the monorepo:
- **Preference Learning** algorithms for user matching
- **Scoring Systems** for compatibility assessment
- **Analytics Collection** for model improvement
- **Real-time Matching** with Redis caching
- **Model Training** scripts and pipelines

### 🏗️ **Architecture Benefits**
The monorepo provides:
- **Unified CI/CD** for all Meetinity services
- **Cross-service ML** integration and data sharing
- **Consistent model deployment** and versioning
- **Centralized analytics** and monitoring
- **Simplified dependency** management for ML libraries

---

**📅 Archived on:** September 29, 2025  
**🔗 Monorepo:** https://github.com/decarvalhoe/meetinity  
**📧 Questions:** Please open issues in the monorepo

---

## 📋 **Original Service Description**

The Meetinity Matching Service provided intelligent user matching using machine learning algorithms, swipe-based interactions, and real-time scoring systems to connect professionals based on compatibility and shared interests.

**Key features now available in the monorepo:**
- User Matching Algorithm with professional profile analysis
- Swipe Functionality (Tinder-like interface)
- Profile Suggestions with personalized recommendations
- Match Detection and real-time notifications
- Compatibility Scoring based on multiple factors
- Machine Learning model training and inference
- Preference learning from user interactions
