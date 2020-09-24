import pandas as pd
import logging
import yaml



import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def run_load_data(args):
        logger.info("Downloading train data to %s", args.config['data']['train_local'])
        df_train = pd.read_csv(args.config['data']['train_url'])
        df_train.to_csv(args.config['data']['train_local'], index=None)
        
        logger.info("Downloading test data to %s", args.config['data']['test_local'])        
        df_test = pd.read_csv(args.config['data']['test_url'])
        df_test.to_csv(args.config['data']['test_local'], index=None)
        
        logger.info("Data downloads finished")



