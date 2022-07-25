from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.model import general_recommender

from newmodel import NewModel
from newtrainer import NewTrainer

from recbole.config import Config
from recbole.data import create_dataset, data_preparation


parameter_dict = {
    'embedding_size': 64
}

if __name__ == '__main__':

    config = Config(model=NewModel, dataset='ml-100k', config_dict=parameter_dict)
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
    model = general_recommender.BPR(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = NewTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))