from FL_Hunter.config import FL_HunterConfig

from FL_Hunter.models.get_model import get_RF_model
from FL_Hunter.workflow import train_exp_sklearn_classifier

if __name__ == '__main__':
    train_exp_sklearn_classifier(FL_HunterConfig().parse_args(), get_RF_model)
