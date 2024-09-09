
# Symbol Detection on Historical Maps
This folder allows you to train/finetune models for symbol detection on old maps, primarily focusing on Cassini maps. Here, you will find all the processes to prepare and augment your data, as well as train, evaluate, and test your models.
## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
## Project Structure
The folder is organized as follows:
```plaintext
├── notebooks/                      # Folder to run notebooks 
│   ├── utils/                      # Utilities functions for the notebooks
│   ├── notebook1.ipynb
│   └── ...
├── models/                         # Folder containing modules for the training of different models
│   ├── detectron2/
│   ├── detr/
│   └── yolo/
├── datasets/                       # Folder containing modules to prepare/augment the datasets
├── data/                           # Folder containing the datasets
├── outputs/                        # Folder with the weights of the different models
├── runs/                           # Folder with training/validation logs
├── tests/                          # Folder for testing different implementations
├───── requirements.txt
└───── README.md
```

## Getting Started
### Prerequisites
Make sure you have the following installed:
- Python 3.7 or higher
- Git
- Virtual environment (optional but recommended)
### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
## Usage
This part is supposed to be used through the `notebooks/` folder. You can know more about the process to follow [here](notebooks/Pipeline.md).
### Implementations
Most of the implementation scripts are used in :
- `datasets/` where the modules to manipulate or visualize the datas are stored.
- `models/` where the scripts to implement and instantiate the models are stored.
- `notebooks/utils` utilities functions used by the notebooks.
### Datasets
The datasets for training are stored in the `datas/` folder and not the `datasets/` one !
It is organized as :
```
models/                        
│   ├── coco_datasets/      #COCO formated datasets
└── └── yolo_datasets/      #YOLO formated datasets
```
### Outputs
The `outputs/` directory will contain the weights of the different models after training. You can use these weights for further evaluation or deployment.
### Logs
Training and validation logs are stored in the `runs/` directory. You can review these logs to monitor the performance and progress of your training sessions.
## Contact
Charles Sutty - charles.m.sutty@gmail.com
Project Link: [https://github.com/your_username/your_repo](https://github.com/your_username/your_repo)
