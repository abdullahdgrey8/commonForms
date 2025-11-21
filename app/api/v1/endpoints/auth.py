from datetime import timedelta
from fastapi import APIRouter, HTTPException, status
from app.models.auth import LoginRequest, Token
from app.core.config import settings
from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Hardcoded user database (replace with real database in production)
HARDCODED_HASH = "$2b$12$YtuJWpggaeY/ap.s0sZnOefFJ1CfTFJhMZRdptXmDyrkXeWsDR2nC"

USERS_DB = {
    settings.TEST_USER_USERNAME: {
        "username": settings.TEST_USER_USERNAME,
        "hashed_password": HARDCODED_HASH,  # ← Hardcoded, safe for dev
        "disabled": False
    }
}

def authenticate_user(username: str, password: str) -> dict | None:
    """
    Authenticate a user by username and password
    
    Args:
        username: User's username
        password: User's plain password
        
    Returns:
        User dict if authentication successful, None otherwise
    """
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


@router.post("/login", response_model=Token)
async def login(credentials: LoginRequest):
    """
    Authenticate user and return JWT token
    
    **Test Credentials:**
    - Username: `admin`
    - Password: `changeme123`
    
    **Returns:**
    - `access_token`: JWT token for authentication
    - `token_type`: Always "bearer"
    - `expires_in`: Token expiration time in seconds
    """
    logger.info(f"Login attempt for user: {credentials.username}")
    
    # Authenticate user
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        logger.warning(f"Failed login attempt for: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    logger.info(f"Successful login for user: {credentials.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )