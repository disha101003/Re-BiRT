hackathonF23-artix
==============================

Reproducing BiRT Architecture (Bio-inspired replay for transformers in Continual Learning)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    |   |__ViT_CiFAR10_data <- CiFAR10 dataset
    |   |__VIT_CiFAR100_data <- CiFAR100 dataset
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |   |__models_CiFAR10 <- model_g, model_f_w, model_f_s weights for CiFAR10
    |   |   |__model_g <- model_f_s weights for CiFAR100
    |   |   |__model_f_s <- model_f_s weights for CiFAR100
    |   |   |__model_f_w <- model_f_w weights for CiFAR100 
    |   |__models_CiFAR100  <- model_g, model_f_w, model_f_s weights for CiFAR100
    |       |__model_g <- model_f_s weights for CiFAR100
    |       |__model_f_s <- model_f_s weights for CiFAR100
    |       |__model_f_w <- model_f_w weights for CiFAR100 
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── dataset.py           <- Scripts to download or generate data
    |   |__ const.py           <- contains all the constants
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── arch.py     <-  architecture code of BiRT
    │   │   └── train.py    <- training code for BiRT for all tasks
    │   │   └── train_finetune_balanced_dataset.py    <- finetuning model on balanced dataset
    |   |   |__ test.py       <- testing code for architecture
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

This code implements BiRT architecture for continual learning. 
The paper was implemented using V100 GPUs with 32GB GPU RAM This implementation was trained and tested on CiFAR 10 and CiFAR 100.
The python environment used is Python 3. 
To run this implementation, run src.models.train first, for training the model for each tasks, run src.models.train_finetune_balanced_dataset, to fine tune the model on after training on all tasks, and to test the model run src.models.test 

Therefore the command sequence looks like this. Change the const.py file in src/models according to the experiment you wish to run. See comments in const.py for specific information.

To test the ViT code when split into g() and f(), set FINE_TUNE_SIZE to be size of training dataset i.e 50,000. The original ViT code can be tested from https://github.com/tintn/vision-transformer-from-scratch/blob/main/vision_transformers.ipynb by changing the dataset accordingly.

python -m src.models.train
python -m src.models.test
python -m src.models.train_finetune_balanced_dataset
python -m src.models.test

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
