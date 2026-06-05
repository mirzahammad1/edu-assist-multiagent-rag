"""
utils/db_sessions.py
Manages Rate Limiting and Admin Sessions using MongoDB TTL indexes.
"""
import secrets
import datetime
import logging
from pymongo import MongoClient
from config.settings import MONGO_URI, DB_NAME
from utils.db import get_db_client

logger = logging.getLogger("eduassist.db_sessions")

_client = None
_db = None
_sessions_col = None
_rate_col = None

def _init_db():
    global _client, _db, _sessions_col, _rate_col
    if _client is None:
        try:
            _client = get_db_client()
            if not _client: return
            _db = _client[DB_NAME]
            
            _sessions_col = _db["admin_sessions"]
            _sessions_col.create_index("expireAt", expireAfterSeconds=0)
            
            _rate_col = _db["rate_limits"]
            _rate_col.create_index("expireAt", expireAfterSeconds=0)
            _rate_col.create_index("session_id")
        except Exception as e:
            logger.error("Failed to connect to MongoDB for sessions/rate limits: %s", e)

def create_admin_session(ttl_seconds: int) -> str:
    _init_db()
    token = secrets.token_hex(32)
    if _sessions_col is not None:
        expire_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=ttl_seconds)
        try:
            _sessions_col.insert_one({"token": token, "expireAt": expire_at})
        except Exception as e:
            logger.error("Failed to create admin session: %s", e)
    return token

def is_valid_session(token: str) -> bool:
    if not token: 
        return False
    _init_db()
    if _sessions_col is not None:
        try:
            doc = _sessions_col.find_one({"token": token})
            if doc and doc.get("expireAt") and doc["expireAt"].replace(tzinfo=datetime.timezone.utc) > datetime.datetime.now(datetime.timezone.utc):
                return True
        except Exception as e:
            logger.error("Failed to validate admin session: %s", e)
    return False

def is_rate_limited(session_id: str, limit: int, window_seconds: int = 60) -> bool:
    _init_db()
    if _rate_col is None:
        return False # Fail open if DB is down
        
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(seconds=window_seconds)
    
    try:
        count = _rate_col.count_documents({
            "session_id": session_id,
            "timestamp": {"$gte": cutoff}
        })
        
        if count >= limit:
            return True
            
        _rate_col.insert_one({
            "session_id": session_id,
            "timestamp": now,
            "expireAt": now + datetime.timedelta(seconds=window_seconds)
        })
    except Exception as e:
        logger.error("Failed to check rate limit: %s", e)
        
    return False

def logout_session(token: str):
    if token:
        _init_db()
        if _sessions_col is not None:
            try:
                _sessions_col.delete_one({"token": token})
            except Exception as e:
                pass
