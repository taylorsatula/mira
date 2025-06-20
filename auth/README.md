# Authentication System

Minimal passwordless authentication with TouchID/WebAuthn support.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
# Required
export DATABASE_URL="postgresql://user:pass@localhost/dbname"
export JWT_SECRET_KEY="your-secret-key"
export SMTP_HOST="smtp.gmail.com"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export FROM_EMAIL="noreply@yourdomain.com"

# Optional (have defaults)
export SMTP_PORT="587"
export SMTP_USE_TLS="true"
export WEBAUTHN_RP_ID="yourdomain.com"
export WEBAUTHN_RP_NAME="Your App Name"
export WEBAUTHN_ORIGIN="https://yourdomain.com"
export APP_URL="https://yourdomain.com"
```

3. Include in your FastAPI app:
```python
from fastapi import FastAPI
from auth.api import router as auth_router

app = FastAPI()
app.include_router(auth_router)
```

## API Endpoints

### Passwordless (Magic Link)
- `POST /auth/request-magic-link` - Send magic link email
- `POST /auth/verify-magic-link` - Verify token and get session

### WebAuthn/TouchID
- `POST /auth/webauthn/register/start` - Start TouchID registration
- `POST /auth/webauthn/register/finish` - Complete registration
- `POST /auth/webauthn/authenticate/start` - Start TouchID login
- `POST /auth/webauthn/authenticate/finish` - Complete login

### Session Management
- `GET /auth/me` - Get current user
- `POST /auth/logout` - Invalidate session

## Usage Example

```python
from auth import get_current_user

# In your endpoints
@app.get("/protected")
async def protected_route(user: Dict = Depends(get_current_user)):
    return {"message": f"Hello {user['email']}"}
```

## Features

- **Passwordless**: Email magic links (10 min expiry)
- **TouchID/FaceID**: WebAuthn platform authenticators
- **Rate Limiting**: 5 requests per 5 minutes per email
- **Multi-tenant**: Optional tenant_id support
- **Sessions**: JWT with 7-day expiry
- **Simple**: ~800 lines total vs 7,760 original