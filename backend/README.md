# Backend API Documentation

## Authentication & User System

This app supports two types of users with different recommendation algorithms:

### User Types

#### 🆕 New Users (Auth Flow)
- **Created via**: `/auth/signup` endpoint
- **User IDs**: 27-digit numeric strings (e.g., `"123456789012345678901234567"`)
- **Recommendations**: Real-time ML inference using user attributes
- **Data**: No interaction history (cold start)

#### 📊 Existing Users (Imported from ML Dataset)
- **Created via**: App installation import from Feast feature store
- **User IDs**: Non-27-digit identifiers (e.g., `"alice_2023"`, `"demo_shopper_1"`)
- **Recommendations**: Pre-computed recommendations from interaction history
- **Data**: Rich interaction history in ML feature store

### 🧪 Development & Testing

#### Test Users with Known Passwords
During database seeding, test users are created using **real Feast user IDs** with known credentials.

**📁 Configuration**: Test user settings are defined in `backend/config/test_users.yaml` for easy customization.

**🎯 Key Benefits:**
- ✅ **Known passwords** for easy login
- ✅ **Rich interaction history** from Feast ML dataset
- ✅ **Lightning-fast recommendations** via pre-computed `user_top_k_items`
- ✅ **Authentic ML experience** with real personalization data
- ✅ **Configurable** via YAML - easy to modify or add new test users

**⚠️ Important**: Test users take precedence over Feast imports. If a Feast user exists with the same ID, the Feast import will be skipped to preserve test user credentials.

| Email | Password | User ID | Type | Description |
|-------|----------|---------|------|-------------|
| `demo1@example.com` | `demo123` | *Real Feast ID* | Existing | Electronics & Books enthusiast |
| `demo2@example.com` | `demo123` | *Real Feast ID* | Existing | Sports & Home products |
| `demo3@example.com` | `demo123` | *Real Feast ID* | Existing | Clothing & Books & Electronics |
| `demo4@example.com` | `demo123` | *Real Feast ID* | Existing | Electronics & Sports & Home |
| `demo5@example.com` | `demo123` | *Real Feast ID* | Existing | Books & Electronics |

*User IDs are dynamically selected from the Feast dataset during seeding. To customize test users, edit `backend/config/test_users.yaml`.*



### 🔄 Recommendation Endpoints

#### For Existing Users
```http
GET /recommendations/{user_id}
```
Fast lookup of pre-computed recommendations.

#### For New Users
```http
POST /recommendations
Authorization: Bearer {token}
Content-Type: application/json

{
  "num_recommendations": 10
}
```
Real-time ML inference using user attributes.

### 🚀 Getting Started

1. **Initialize Database:**
   ```bash
   python init_backend.py
   ```
   This creates tables and imports users from Feast.

2. **Test with Existing User:**
   ```bash
   # Login with known password (any demo1-5 user)
   curl -X POST http://localhost:8000/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email": "demo1@example.com", "password": "demo123"}'
   ```

3. **Test with New User:**
   ```bash
   curl -X POST http://localhost:8000/auth/signup \
     -H "Content-Type: application/json" \
     -d '{"email": "newuser@example.com", "password": "mypass", "age": 25, "gender": "Other"}'
   ```

4. **Get Recommendations:**
   ```bash
   # Use the token from login/signup response
   curl -H "Authorization: Bearer {your_token}" \
     http://localhost:8000/recommendations
   ```

### 🔒 Security

- **Authentication**: JWT tokens with 24-hour expiry
- **Password Hashing**: Bcrypt with secure salt rounds
- **Test Users**: Known passwords for development (configurable via YAML)
- **Token Validation**: Server-side validation via `/auth/me` endpoint

This setup allows you to experience both the "new user cold start" flow and the "existing user with rich data" flow with standard authentication!
