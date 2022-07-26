from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.model import general_recommender
from sklearn.model_selection import KFold
from newmodel import NewModel
from newtrainer import NewTrainer

from recbole.config import Config
from recbole.data import create_dataset, data_preparation


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
    'valid_metric': 'MRR@5',
    'eval_args':{'split': {'LS': 'valid_and_test'}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}, #{'RS': [0.8,0.1,0.1]}
    'topk': 5 
    #load_split_dataloaders
}

if __name__ == '__main__':

    config = Config(model=NewModel, dataset='ml-100k',
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
    model = NewModel(config, train_data.dataset).to(config['device'])

    logger.info(model)

    # trainer loading and initialization
    trainer = NewTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))
