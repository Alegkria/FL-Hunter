from FL_Hunter.config import FL_HunterConfig
from FL_Hunter.models import get_GAT_model
from FL_Hunter.workflow import train_exp_CFL

if __name__ == '__main__':
    # logger.info("Disable JIT because of DGL")
    # set_jit_enabled(False)
    train_exp_CFL(FL_HunterConfig().parse_args(), get_GAT_model, plot_model=False)
