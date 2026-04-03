import os
import uuid
import hashlib
import hmac
from datetime import datetime, timedelta, date, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

router = APIRouter()

_HASH_SALT_PREFIX = "stellar-golf-pw-"


def _hash_password(password: str, salt: str | None = None) -> str:
    """Hash password using PBKDF2-SHA256 (no bcrypt dependency)."""
    if salt is None:
        salt = uuid.uuid4().hex[:16]
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), (salt + _HASH_SALT_PREFIX).encode(), 200_000)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt = stored_hash.split("$")[0]
    return hmac.compare_digest(_hash_password(password, salt), stored_hash)
security = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "JWT_SECRET not set — authentication will fail. "
        "Set JWT_SECRET in Render environment variables."
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 72

users_db: dict[str, dict] = {}


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ProLoginRequest(BaseModel):
    email: str
    password: str
    invite_code: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user_id: str
    email: str
    is_pro: bool


class GuestResponse(BaseModel):
    token: str
    user_id: str
    is_guest: bool
    remaining_analyses: int


def create_token(user_id: str, email: str, is_pro: bool, is_guest: bool = False) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "is_pro": is_pro,
        "is_guest": is_guest,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials) -> dict:
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    if credentials is None:
        return None
    try:
        return verify_token(credentials)
    except HTTPException:
        return None


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    for uid, user in users_db.items():
        if user["email"] == req.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    password_hash = _hash_password(req.password)

    users_db[user_id] = {
        "id": user_id,
        "email": req.email,
        "password_hash": password_hash,
        "is_pro": False,
        "daily_count": 0,
        "last_reset": date.today().isoformat(),
        "created_at": datetime.utcnow().isoformat(),
    }

    token = create_token(user_id, req.email, is_pro=False)
    return AuthResponse(token=token, user_id=user_id, email=req.email, is_pro=False)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    user = None
    for uid, u in users_db.items():
        if u["email"] == req.email:
            user = u
            break

    if user is None or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    today = date.today().isoformat()
    if user["last_reset"] != today:
        user["daily_count"] = 0
        user["last_reset"] = today

    token = create_token(user["id"], user["email"], user["is_pro"])
    return AuthResponse(
        token=token, user_id=user["id"], email=user["email"], is_pro=user["is_pro"]
    )


@router.post("/pro-login", response_model=AuthResponse)
async def pro_login(req: ProLoginRequest):
    VALID_INVITE_CODES = {"STELLAR2024", "GOLFPRO2024", "PREMIUM2024"}

    user = None
    for uid, u in users_db.items():
        if u["email"] == req.email:
            user = u
            break

    if user is None:
        if req.invite_code and req.invite_code in VALID_INVITE_CODES:
            user_id = str(uuid.uuid4())
            password_hash = _hash_password(req.password)
            users_db[user_id] = {
                "id": user_id,
                "email": req.email,
                "password_hash": password_hash,
                "is_pro": True,
                "daily_count": 0,
                "last_reset": date.today().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
            }
            user = users_db[user_id]
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials or invite code")
    else:
        if not _verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid password")

        if req.invite_code and req.invite_code in VALID_INVITE_CODES:
            user["is_pro"] = True

        if not user["is_pro"]:
            raise HTTPException(
                status_code=403,
                detail="This account does not have Pro access. Please provide a valid invite code.",
            )

    token = create_token(user["id"], user["email"], is_pro=True)
    return AuthResponse(
        token=token, user_id=user["id"], email=user["email"], is_pro=True
    )


@router.post("/guest", response_model=GuestResponse)
async def guest_login():
    guest_id = f"guest_{uuid.uuid4().hex[:12]}"
    token = create_token(guest_id, "guest@stellar.ai", is_pro=False, is_guest=True)
    return GuestResponse(
        token=token, user_id=guest_id, is_guest=True, remaining_analyses=3
    )
