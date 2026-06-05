"""
utils/db.py

Shared MongoDB client utility to consolidate multiple independent connection pools.
"""
import logging
from pymongo import MongoClient
from config.settings import MONGO_URI, DB_NAME

logger = logging.getLogger("eduassist.db")

_client = None

def get_db_client():
    global _client
    if _client is None:
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            logger.info("MongoDB shared client initialized.")
        except Exception as e:
            logger.error("Failed to connect to MongoDB: %s", e)
    return _client

def get_db(db_name=DB_NAME):
    client = get_db_client()
    return client[db_name] if client else None
