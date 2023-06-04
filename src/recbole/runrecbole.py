import recbole

# Load the dataset
dataset = recbole.DataLoader('.inter',
                                     config_file='/home/eduardo/masters/MovieLens-100k/src/recbole/config/config.yml',
                                     train_file='/home/eduardo/masters/MovieLens-100k/data/u1/u1_train.inter',
                                     valid_file='/home/eduardo/masters/MovieLens-100k/data/u1/u1_valid.inter',
                                     test_file='/home/eduardo/masters/MovieLens-100k/data/u1/u1_test.inter')

# Initialize the sequential model
model = recbole.model.SeqRecModel(dataset)

# Set up logger
logger = recbole.utils.get_logger()

# Set up trainer
trainer = recbole.trainer.Trainer(model, dataset)

# Train the model
trainer.train()

# Evaluate the model
evaluate_result = trainer.evaluate()

# Print the evaluation result
logger.info(evaluate_result)