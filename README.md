# Template_Pytorch_Parallel


## Utils
utils.py contains Python functions to setup Distributed Training of Pytorch Models, utilities such as create directory on main process and initiate signal for relaunching slurm jobs

## Logger
The logger class (logger.py) is a simple logger with an API similar to wandb (init, log and finish), it can be selected to either directly log to wandb or to a txt file with a json format.
This logger is useful when dealing with compute clusters not connected to internet and with compute sessions limited in time (wandb can not continue an offline job)
It is possible to upload the txt file to wandb by using python3 logger.py --path folder_of_logs__or__log_file

## Models
The models directory is based on torchvision models dir, implementation exemple in resnet.py