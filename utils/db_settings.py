"""
utils/db_settings.py

Manages dynamic app settings in MongoDB, allowing Admin UI to toggle agents
and edit smart fallbacks without code changes.
"""
from pymongo import MongoClient
import logging
from typing import Dict, Any

from config.settings import MONGO_URI, DB_NAME

logger = logging.getLogger("eduassist.settings.db")

_client = None
_db = None
_collection = None

# Default settings seed
DEFAULT_SETTINGS = {
    "agents": {
        "faq": True,
        "webscraper": True,
        "vectordb": True,
        "consistency": True
    },
    "smart_fallbacks": [
        {
            "keywords": ["timetable", "time table", "class schedule", "lecture schedule"],
            "response": {
                "message": "Class timetables are available on the Student LMS Portal.",
                "link_label": "🖥 LMS Portal",
                "link_url": "https://lms.kiet.edu.pk/kietlms/my/Student_Portal.php"
            }
        },
        {
            "keywords": ["complaint", "complain", "grievance", "report issue", "problem with"],
            "response": {
                "message": "For complaints or grievances, please contact the Admin Office directly.",
                "link_label": "📞 Departments Contact",
                "link_url": "https://kiet.edu.pk/departments-contact/"
            }
        },
        {
            "keywords": ["hostel", "dormitory", "accommodation", "room", "boarding"],
            "response": {
                "message": "For hostel availability and room allocation, contact Student Affairs.",
                "link_label": "📞 Departments Contact",
                "link_url": "https://kiet.edu.pk/departments-contact/"
            }
        },
        {
            "keywords": ["job", "vacancy", "career", "hiring", "recruitment", "apply for job"],
            "response": {
                "message": "Current job openings at KIET are listed on the Jobs page.",
                "link_label": "💼 Jobs at KIET",
                "link_url": "https://kiet.edu.pk/jobs/"
            }
        },
        {
            "keywords": ["lost", "found", "missing item", "lost item"],
            "response": {
                "message": "For lost and found items, please visit the Security Office on campus.",
                "link_label": "📞 Departments Contact",
                "link_url": "https://kiet.edu.pk/departments-contact/"
            }
        }
    ],
    "followup_map": [
        {
            "keywords": ["admission", "apply", "eligibility", "requirement", "merit", "entry test"],
            "suggestions": [
                "What documents are required?",
                "What is the fee structure?",
                "When is the last date to apply?"
            ]
        },
        {
            "keywords": ["fee", "fees", "tuition", "payment", "cost"],
            "suggestions": [
                "Is there any scholarship available?",
                "What is the admission process?",
                "Are there any fee discounts?"
            ]
        },
        {
            "keywords": ["scholarship", "discount", "financial aid", "bursary"],
            "suggestions": [
                "What is the fee structure?",
                "What is the eligibility for scholarship?",
                "How do I apply for scholarship?"
            ]
        },
        {
            "keywords": ["program", "degree", "bachelor", "master", "bs", "ms", "course"],
            "suggestions": [
                "What is the fee structure?",
                "What is the admission process?",
                "What departments does KIET have?"
            ]
        },
        {
            "keywords": ["hostel", "dormitory", "accommodation"],
            "suggestions": [
                "What is the hostel fee?",
                "Is hostel available for girls?",
                "How do I apply for hostel?"
            ]
        },
        {
            "keywords": ["transport", "bus", "shuttle", "route"],
            "suggestions": [
                "What is the transport fee?",
                "Which areas does KIET transport cover?",
                "How do I register for transport?"
            ]
        },
        {
            "keywords": ["event", "code jung", "hackathon", "fest", "seminar", "workshop"],
            "suggestions": [
                "How do I register for the event?",
                "What are the prizes?",
                "Who can participate?"
            ]
        },
        {
            "keywords": ["lms", "portal", "login", "password"],
            "suggestions": [
                "How do I reset my LMS password?",
                "Where can I see my timetable?",
                "How do I check my results?"
            ]
        }
    ]
}

def get_collection():
    global _client, _db, _collection
    if _client is None:
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            _db = _client[DB_NAME]
            _collection = _db["app_settings"]
        except Exception as e:
            logger.error("Failed to connect to MongoDB for settings: %s", e)
            return None
    return _collection

def get_app_settings() -> Dict[str, Any]:
    coll = get_collection()
    if coll is None:
        return DEFAULT_SETTINGS
    
    doc = coll.find_one({"_id": "global_settings"})
    if not doc:
        # Seed the database
        doc = {"_id": "global_settings", **DEFAULT_SETTINGS}
        try:
            coll.insert_one(doc)
        except Exception as e:
            logger.error("Failed to insert default settings: %s", e)
    
    if "_id" in doc:
        del doc["_id"]
    return doc

def update_app_settings(updates: Dict[str, Any]) -> bool:
    coll = get_collection()
    if coll is None:
        return False
    try:
        coll.update_one(
            {"_id": "global_settings"},
            {"$set": updates},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error("Failed to update settings: %s", e)
        return False

# In-memory cache to avoid hitting DB on every request
_cached_settings = None

def get_cached_settings() -> Dict[str, Any]:
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = get_app_settings()
    return _cached_settings

def invalidate_settings_cache():
    global _cached_settings
    _cached_settings = None
