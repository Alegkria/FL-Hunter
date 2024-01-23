from FL_Hunter.config import FL_HunterConfig
from FL_Hunter.models.get_model import get_DNN_model
from FL_Hunter.workflow import train_exp_CFL

if __name__ == '__main__':
    train_exp_CFL(FL_HunterConfig().parse_args(), get_DNN_model, plot_model=False)
