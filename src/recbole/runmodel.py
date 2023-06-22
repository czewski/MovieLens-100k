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
from recbole.trainer import Trainer

import csv   
from datetime import datetime

import matplotlib.pylab as plt


## https://recbole.io/docs/get_started/started/sequential.html
## Aqui tem bastante informação de como podemos alterar o dataset para ser sequencial

## https://recbole.io/docs/user_guide/usage/use_tensorboard.html
## Informação de como usar o tensorboard (dados de treinamento e validação)


## https://www.tensorflow.org/recommenders/examples/sequential_retrieval

## https://ceur-ws.org/Vol-2955/paper8.pdf ------ PAPER MT BOM QUE TEM USO DE SEQ RECOMMENDATION COM MOVIELENS
## "The MovieLens datasets are well established for evaluating" -- NESSA PARTE TEM CITAÇÕES


if __name__ == '__main__':
    models = [
    sequential_recommender.STAMP,
    # sequential_recommender.BERT4Rec,
    # sequential_recommender.NARM,
    # sequential_recommender.NPE,
    # sequential_recommender.SASRec,
    #sequential_recommender.TransRec
    #CustomLSTM
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
        config = Config(model=model, config_file_list=['/home/eduardo/masters/MovieLens-100k/src/recbole/config/configSequentialModels.yml']) #  config_dict=parameter_dict
        #https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87
           
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
        model = model(config, dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = Trainer(config, model) #NewTrainer(config, model)

        val_metric_values = []

        def val_callback(epoch_idx, valid_score):
            val_metric_values.append(valid_score)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data,callback_fn=val_callback, show_progress=True) #config['show_progress']

        # model evaluation
        test_result = trainer.evaluate(test_data)
        model_unique_id = datetime.now().strftime("%Y-%m-%d-%H:%M:%S_") + model.__class__.__name__
    
        print(trainer.train_loss_dict)

        fig1, ax1 = plt.subplots()
        ax1.plot(list(trainer.train_loss_dict.keys()), list(trainer.train_loss_dict.values()))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Value')
        ax1.set_title('Loss Plot')
        fig1.savefig('loss_curve'+model_unique_id+'.png')

        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.title('Loss Plot')
        # plt.plot(list(trainer.train_loss_dict.keys()), list(trainer.train_loss_dict.values()))
        # plt.savefig('loss_curve'+model_unique_id+'.png')

        fig2, ax2 = plt.subplots()
        ax2.plot(val_metric_values[::-1])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.set_title('Valid Plot')
        fig2.savefig('valid_curve'+model_unique_id+'.png')

        # print(val_metric_values)
        # plt.plot(val_metric_values)
        # plt.xlabel('Epoch')
        # plt.ylabel('Value')
        # plt.title('Valid Plot')
        # plt.show()
        # plt.savefig('valid_curve'+model_unique_id+'.png')

        logger.info('---------------------------------')
        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))
        logger.info('model name: {}'.format(model_unique_id))
        logger.info('---------------------------------')

       # best_valid_result['MRR'] # testar se isso funciona
        fields=[model.__class__.__name__,config['dataset'],datetime.now(), best_valid_result,test_result]  #log location, model location and name
        with open(r'data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
