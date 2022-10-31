from logging import getLogger
from recbole.utils import init_logger, init_seed
#from recbole.trainer import Trainer
from recbole.model import general_recommender, sequential_recommender
#from sklearn.model_selection import KFold
#from recbole.custom_model_general import NewModel
from newtrainer import NewTrainer
from custom_model_sequential import CustomLSTM

from recbole.config import Config
from recbole.data import create_dataset, data_preparation

import csv   
from datetime import datetime

if __name__ == '__main__':
    models = [
    CustomLSTM
    #sequential_recommender.
    #sequential_recommender.STAMP,
    # NewModel,
    # general_recommender.Pop, 
    # general_recommender.ItemKNN, 
    # general_recommender.BPR, 
    # general_recommender.ConvNCF,
    # general_recommender.MultiDAE, 
    # general_recommender.MultiVAE,
    # general_recommender.CDAE
    ]

    for model in models: 
        config = Config(model=model,config_file_list=['/home/eduardo/projects/MovieLens-100k/src/recbole/config/configSequentialModels.yml']) #config_dict=parameter_dict
        #https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87
        #o que muda em um modelo normal pra um modelo de recsys?

        init_seed(config['seed'], config['reproducibility'])
        
        # logger initialization
        init_logger(config)
        logger = getLogger()
        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        model = model(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = NewTrainer(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=config['show_progress'])

        # model evaluation
        test_result = trainer.evaluate(test_data)

        logger.info('---------------------------------')
        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))
        logger.info('model name: {}'.format(model.__class__.__name__))


       # best_valid_result['MRR'] # testar se isso funciona
        fields=[model.__class__.__name__,config['dataset'],datetime.now(), best_valid_result,test_result]  #log location, model location and name
        with open(r'data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
