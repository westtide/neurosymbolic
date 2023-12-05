import logging
import time

timestamp = time.strftime('%Y%m%d%H%M%S')
logger = logging.getLogger('start: log')
handler = logging.FileHandler(f'log/{timestamp}.log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
