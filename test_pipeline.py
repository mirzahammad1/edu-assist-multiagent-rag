import logging
logging.basicConfig(level=logging.INFO)
import traceback
from pipeline import get_bot

try:
    bot = get_bot()
    print('SUCCESS')
except Exception as e:
    print('ERROR:')
    traceback.print_exc()
