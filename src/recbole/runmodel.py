from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.model import general_recommender
from sklearn.model_selection import KFold
from newmodel import NewModel
from newtrainer import NewTrainer

from recbole.config import Config
from recbole.data import create_dataset, data_preparation

import csv   
from datetime import datetime
from sklearn.model_selection import KFold

parameter_dict = {
    'seed': 1,
    'reproducibility': True,
    'embedding_size': 64,
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'TIME_FIELD': 'timestamp',
    'shuffle': False,  # Whether or not to shuffle the training data before each epoch

    # Train
    'epochs': 50,
    'learner': 'adam',  # ['adam', 'sgd', 'adagrad', 'rmsprop', 'sparse_adam']
    'learning_rate': 0.001,
    'train_batch_size': 2048,


    #Evaluation
    'metrics':['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'MAP', 'TailPercentage'], # 'AUC', 'MAE', 'RMSE'
    'valid_metric': 'MRR@10',
    'eval_args':{'split': {'RS': [0.8,0.1,0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}, #{'RS': [0.8,0.1,0.1]} {'LS': 'valid_and_test'}
    'topk': 10 
    #load_split_dataloaders
}

if __name__ == '__main__':
    models = [general_recommender.Pop, general_recommender.ItemKNN, general_recommender.BPR, general_recommender.ConvNCF, general_recommender.MultiDAE, general_recommender.MultiVAE, general_recommender.CDAE]

    for model in models: 
        
        config = Config(model=model, dataset='ml-100k',
                        config_dict=parameter_dict)
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
         #model = general_recommender.BPR(config, train_data.dataset).to(config['device'])
        model = model(config, train_data.dataset).to(config['device'])

        logger.info(model)

        # trainer loading and initialization
        trainer = NewTrainer(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

        # model evaluation
        test_result = trainer.evaluate(test_data)

        logger.info('---------------------------------')
        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))
        logger.info('model name: {}'.format(model.__class__.__name__))


        fields=[model.__class__.__name__,best_valid_result,test_result, datetime.now()]
        with open(r'data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
