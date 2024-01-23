from sklearn.svm import LinearSVC

from FL_Hunter.config import FL_HunterConfig
from FL_Hunter.workflow import train_exp_sklearn_classifier
from FL_Hunter.models.get_model import ClassifierProtocol
from failure_dependency_graph import FDG


def get_model(cdp: FDG, config: FL_HunterConfig) -> ClassifierProtocol:
    return LinearSVC(probability=True)


if __name__ == '__main__':
    train_exp_sklearn_classifier(FL_HunterConfig().parse_args(), get_model)
