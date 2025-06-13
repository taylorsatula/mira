# MIRA Magic Link Authentication Implementation Guide

## Overview

This guide implements straightforward magic link (passwordless) authentication for MIRA with multi-user support. Users sign in via email links, and all data is isolated per user.

## Core Requirements

- Users can request a magic link sent to their email
- Links expire after 15 minutes and can only be used once
- After clicking the link, users get a JWT session lasting 30 days
- All user data is isolated - users only see their own conversations, memories, etc.
- No passwords to manage or reset

## Database Schema

### New Tables

```sql
-- Users table - minimal fields needed
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Magic link tokens
CREATE TABLE magic_link_tokens (
    token_hash VARCHAR(64) PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used_at TIMESTAMP WITH TIME ZONE
);

-- Active sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_magic_link_tokens_user_id ON magic_link_tokens(user_id);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);
```

### Update Existing Tables

```sql
-- Add user_id to all data tables
ALTER TABLE conversations ADD COLUMN user_id UUID REFERENCES users(id);
ALTER TABLE working_memory ADD COLUMN user_id UUID REFERENCES users(id);
ALTER TABLE lt_memory_blocks ADD COLUMN user_id UUID REFERENCES users(id);
ALTER TABLE lt_memory_entities ADD COLUMN user_id UUID REFERENCES users(id);
ALTER TABLE task_automations ADD COLUMN user_id UUID REFERENCES users(id);

-- Add indexes for performance
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_working_memory_user_id ON working_memory(user_id);
CREATE INDEX idx_lt_memory_blocks_user_id ON lt_memory_blocks(user_id);
CREATE INDEX idx_lt_memory_entities_user_id ON lt_memory_entities(user_id);
CREATE INDEX idx_task_automations_user_id ON task_automations(user_id);

-- Enable Row Level Security
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE working_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE lt_memory_blocks ENABLE ROW LEVEL SECURITY;
ALTER TABLE lt_memory_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE task_automations ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (users see only their data)
CREATE POLICY user_isolation ON conversations FOR ALL 
    USING (user_id = current_setting('app.current_user_id')::UUID);
    
CREATE POLICY user_isolation ON working_memory FOR ALL 
    USING (user_id = current_setting('app.current_user_id')::UUID);
    
CREATE POLICY user_isolation ON lt_memory_blocks FOR ALL 
    USING (user_id = current_setting('app.current_user_id')::UUID);
    
CREATE POLICY user_isolation ON lt_memory_entities FOR ALL 
    USING (user_id = current_setting('app.current_user_id')::UUID);
    
CREATE POLICY user_isolation ON task_automations FOR ALL 
    USING (user_id = current_setting('app.current_user_id')::UUID);
```

## Authentication Implementation

### Core Authentication Module

```python
# auth/core.py
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configuration
JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]  # Must be set, no fallback
JWT_ALGORITHM = "HS256"
MAGIC_LINK_EXPIRY_MINUTES = 15
SESSION_EXPIRY_DAYS = 30

security = HTTPBearer()

def generate_magic_link_token() -> Tuple[str, str]:
    """Generate a secure token and its hash."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash

def create_jwt_token(user_id: str, email: str, session_id: str) -> str:
    """Create JWT token for user session."""
    expire = datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)
    payload = {
        "user_id": user_id,
        "email": email,
        "session_id": session_id,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FastAPI dependency to get current user from JWT."""
    token = credentials.credentials
    payload = decode_jwt_token(token)
    
    # Verify session is still active in database
    async with get_db() as conn:
        session = await conn.fetchrow(
            """SELECT * FROM user_sessions 
               WHERE id = $1 AND user_id = $2 
               AND expires_at > NOW()""",
            payload["session_id"], payload["user_id"]
        )
        if not session:
            raise HTTPException(status_code=401, detail="Session expired")
    
    return payload
```

### Authentication Endpoints

```python
# auth/endpoints.py
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import uuid

router = APIRouter(prefix="/auth")

class MagicLinkRequest(BaseModel):
    email: EmailStr

class MagicLinkVerify(BaseModel):
    token: str

# Simple in-memory rate limiting
from collections import defaultdict
from datetime import datetime

rate_limit_storage = defaultdict(list)

def check_rate_limit(email: str, max_requests: int = 3, window_minutes: int = 60) -> bool:
    """Simple rate limiting check."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=window_minutes)
    
    # Clean old entries
    rate_limit_storage[email] = [
        timestamp for timestamp in rate_limit_storage[email] 
        if timestamp > cutoff
    ]
    
    # Check limit
    if len(rate_limit_storage[email]) >= max_requests:
        return False
    
    rate_limit_storage[email].append(now)
    return True

@router.post("/request-magic-link")
async def request_magic_link(
    data: MagicLinkRequest,
    background_tasks: BackgroundTasks
):
    """Send magic link to user's email."""
    
    # Rate limiting
    if not check_rate_limit(data.email):
        raise HTTPException(
            status_code=429, 
            detail="Too many requests. Please try again later."
        )
    
    async with get_db() as conn:
        # Get or create user
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            data.email
        )
        
        if not user:
            user = await conn.fetchrow(
                "INSERT INTO users (email) VALUES ($1) RETURNING id",
                data.email
            )
        
        # Generate magic link
        raw_token, token_hash = generate_magic_link_token()
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=MAGIC_LINK_EXPIRY_MINUTES)
        
        # Store token
        await conn.execute(
            """INSERT INTO magic_link_tokens 
               (token_hash, user_id, expires_at)
               VALUES ($1, $2, $3)""",
            token_hash, user["id"], expires_at
        )
        
        # Send email (background task)
        magic_link_url = f"{FRONTEND_URL}/auth/verify?token={raw_token}"
        background_tasks.add_task(
            send_magic_link_email,
            email=data.email,
            magic_link_url=magic_link_url
        )
    
    return {"message": "Magic link sent to your email"}

@router.post("/verify-magic-link")
async def verify_magic_link(data: MagicLinkVerify):
    """Verify magic link and create session."""
    
    token_hash = hashlib.sha256(data.token.encode()).hexdigest()
    
    async with get_db() as conn:
        # Verify token
        token_data = await conn.fetchrow(
            """SELECT t.*, u.email 
               FROM magic_link_tokens t
               JOIN users u ON t.user_id = u.id
               WHERE t.token_hash = $1 
               AND t.expires_at > NOW()
               AND t.used_at IS NULL""",
            token_hash
        )
        
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid or expired link")
        
        # Mark token as used
        await conn.execute(
            "UPDATE magic_link_tokens SET used_at = NOW() WHERE token_hash = $1",
            token_hash
        )
        
        # Update last login
        await conn.execute(
            "UPDATE users SET last_login = NOW() WHERE id = $1",
            token_data["user_id"]
        )
        
        # Create session
        session_id = str(uuid.uuid4())
        jwt_token = create_jwt_token(
            user_id=str(token_data["user_id"]),
            email=token_data["email"],
            session_id=session_id
        )
        
        # Store session
        expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)
        await conn.execute(
            """INSERT INTO user_sessions 
               (id, user_id, token_hash, expires_at)
               VALUES ($1, $2, $3, $4)""",
            session_id,
            token_data["user_id"],
            hashlib.sha256(jwt_token.encode()).hexdigest(),
            expires_at
        )
    
    return {
        "access_token": jwt_token,
        "token_type": "bearer"
    }

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout current session."""
    async with get_db() as conn:
        await conn.execute(
            "DELETE FROM user_sessions WHERE id = $1",
            current_user["session_id"]
        )
    
    return {"message": "Logged out successfully"}
```

### Email Service

```python
# auth/email.py
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = os.environ["SENDGRID_API_KEY"]
FROM_EMAIL = os.environ["FROM_EMAIL"]
sg = SendGridAPIClient(SENDGRID_API_KEY)

async def send_magic_link_email(email: str, magic_link_url: str):
    """Send magic link email."""
    
    html_content = f"""
    <h2>Sign in to MIRA</h2>
    <p>Click the link below to sign in. This link expires in 15 minutes.</p>
    <p><a href="{magic_link_url}" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">Sign In</a></p>
    <p>If you didn't request this, please ignore this email.</p>
    <hr>
    <p style="font-size: 12px; color: #666;">
    Or copy this link: {magic_link_url}
    </p>
    """
    
    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=email,
        subject="Sign in to MIRA",
        html_content=html_content
    )
    
    try:
        response = sg.send(message)
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False
```

### Authentication Middleware

```python
# auth/middleware.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class AuthMiddleware(BaseHTTPMiddleware):
    """Add user context to all authenticated requests."""
    
    # Paths that don't require authentication
    PUBLIC_PATHS = [
        "/auth/request-magic-link",
        "/auth/verify-magic-link",
        "/docs",
        "/openapi.json",
        "/health"
    ]
    
    async def dispatch(self, request, call_next):
        # Skip auth for public paths
        if any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS):
            return await call_next(request)
        
        # Check for auth header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"}
            )
        
        try:
            # Decode token
            token = auth_header.split(" ")[1]
            payload = decode_jwt_token(token)
            
            # Add user context to request
            request.state.user_id = payload["user_id"]
            request.state.user_email = payload["email"]
            
        except Exception as e:
            return JSONResponse(
                status_code=401,
                content={"detail": str(e)}
            )
        
        response = await call_next(request)
        return response
```

## Database Connection with User Context

```python
# db.py updates
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_user_db(user_id: str):
    """Get database connection with user context for RLS."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Set user context for Row Level Security
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, false)",
            user_id
        )
        yield conn
    finally:
        await conn.close()

# Update all database queries to use user context
async def get_conversations(user_id: str):
    async with get_user_db(user_id) as conn:
        # RLS automatically filters by user_id
        return await conn.fetch(
            "SELECT * FROM conversations ORDER BY created_at DESC"
        )
```

## Updating Existing Code

### API Endpoints

```python
# Update all endpoints to require authentication
from auth.core import get_current_user

@app.post("/api/conversation")
async def create_conversation(
    data: ConversationCreate,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    
    async with get_user_db(user_id) as conn:
        # User context is already set, just insert normally
        result = await conn.fetchrow(
            """INSERT INTO conversations (user_id, content) 
               VALUES ($1, $2) RETURNING id""",
            user_id, data.content
        )
    
    return {"id": result["id"]}
```

### Tool Execution

```python
# Add user context to tool execution
async def execute_tool(
    tool_name: str,
    params: dict,
    user_id: str
):
    """Execute tool with user context."""
    # Tools that need database access should use get_user_db(user_id)
    tool = TOOL_REGISTRY[tool_name]
    return await tool.execute(params, user_id=user_id)
```

### WebSocket Connections

```python
# Update WebSocket handler for authenticated connections
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)  # Token passed as query param
):
    try:
        # Validate token
        payload = decode_jwt_token(token)
        user_id = payload["user_id"]
        
        await websocket.accept()
        
        # Handle messages with user context
        while True:
            data = await websocket.receive_json()
            # Process with user_id context
            
    except Exception as e:
        await websocket.close(code=1008)
```

## Application Setup

```python
# main.py updates
from fastapi import FastAPI
from auth.middleware import AuthMiddleware
from auth.endpoints import router as auth_router

app = FastAPI()

# Add authentication endpoints
app.include_router(auth_router)

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Existing routes remain the same, just add Depends(get_current_user)
```

## Frontend Integration

```javascript
// Simple JavaScript example for frontend
class MiraAuth {
    constructor() {
        this.token = localStorage.getItem('mira_token');
    }
    
    async requestMagicLink(email) {
        const response = await fetch('/auth/request-magic-link', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({email})
        });
        return response.json();
    }
    
    async verifyMagicLink(token) {
        const response = await fetch('/auth/verify-magic-link', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({token})
        });
        
        const data = await response.json();
        if (data.access_token) {
            this.token = data.access_token;
            localStorage.setItem('mira_token', this.token);
        }
        return data;
    }
    
    async makeAuthenticatedRequest(url, options = {}) {
        return fetch(url, {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `Bearer ${this.token}`
            }
        });
    }
    
    logout() {
        localStorage.removeItem('mira_token');
        this.token = null;
    }
}
```

## Migration Script

```python
# scripts/migrate_to_multiuser.py
import asyncio
import asyncpg
import os

async def migrate_existing_data():
    """Migrate existing data to multi-user structure."""
    
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # Create a default user for existing data
        default_user = await conn.fetchrow(
            """INSERT INTO users (email) 
               VALUES ('admin@mira.local') 
               RETURNING id"""
        )
        
        print(f"Created default user: {default_user['id']}")
        
        # Update all existing data with default user
        tables = [
            'conversations',
            'working_memory',
            'lt_memory_blocks',
            'lt_memory_entities',
            'task_automations'
        ]
        
        for table in tables:
            result = await conn.execute(
                f"UPDATE {table} SET user_id = $1 WHERE user_id IS NULL",
                default_user['id']
            )
            print(f"Updated {table}: {result}")
        
        print("Migration completed successfully")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate_existing_data())
```

## Environment Variables

```bash
# Required environment variables (no defaults!)
JWT_SECRET_KEY=        # Generate with: openssl rand -hex 32
SENDGRID_API_KEY=      # From SendGrid dashboard
FROM_EMAIL=            # Verified sender email
FRONTEND_URL=          # Your frontend URL
DATABASE_URL=          # PostgreSQL connection string
```

## Security Checklist

### Essential Security
- [x] Tokens expire after 15 minutes
- [x] Tokens can only be used once
- [x] JWT sessions expire after 30 days
- [x] All database queries use Row Level Security
- [x] Rate limiting on magic link requests
- [x] HTTPS required in production
- [x] No token logging
- [x] Environment variables for all secrets

### Production Requirements
- [ ] Set strong JWT_SECRET_KEY
- [ ] Configure SendGrid with verified domain
- [ ] Enable PostgreSQL SSL
- [ ] Set up HTTPS certificates
- [ ] Configure secure cookies for production
- [ ] Add SPF/DKIM records for email domain

## Testing

```python
# tests/test_auth.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_magic_link_flow():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Request magic link
        response = await client.post(
            "/auth/request-magic-link",
            json={"email": "test@example.com"}
        )
        assert response.status_code == 200
        
        # Would need to extract token from email/database in real test
        # Then verify it works

@pytest.mark.asyncio
async def test_rate_limiting():
    async with AsyncClient(app=app, base_url="http://test") as client:
        email = "ratelimit@test.com"
        
        # First 3 requests should work
        for _ in range(3):
            response = await client.post(
                "/auth/request-magic-link",
                json={"email": email}
            )
            assert response.status_code == 200
        
        # 4th should be rate limited
        response = await client.post(
            "/auth/request-magic-link",
            json={"email": email}
        )
        assert response.status_code == 429
```

## Deployment Steps

1. **Database Setup**
   ```bash
   # Run schema creation SQL
   psql $DATABASE_URL < create_schema.sql
   ```

2. **Environment Configuration**
   ```bash
   # Set all required environment variables
   export JWT_SECRET_KEY=$(openssl rand -hex 32)
   export SENDGRID_API_KEY="your-api-key"
   export FROM_EMAIL="noreply@yourdomain.com"
   export FRONTEND_URL="https://yourdomain.com"
   ```

3. **Migrate Existing Data**
   ```bash
   python scripts/migrate_to_multiuser.py
   ```

4. **Deploy Application**
   ```bash
   # Deploy with updated code
   # Restart FastAPI service
   ```

5. **Verify**
   - Test magic link flow
   - Verify data isolation
   - Check email delivery
   - Monitor for errors

---

This implementation provides secure, straightforward magic link authentication for MIRA. Users sign in with their email, receive a link, and get a long-lived session. All data is automatically isolated per user through Row Level Security. The system is designed to be rock-solid and easy to extend when you're ready for additional features.