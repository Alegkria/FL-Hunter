from loguru import logger
from pyprof import profile, Profiler

from FL_Hunter.config import FL_HunterConfig
from FL_Hunter.evaluation_metrics import get_evaluation_metrics_dict
from FL_Hunter.workflow import format_result_string
from JSS20.system_graph import GraphLibrary
from failure_dependency_graph import FDGModelInterface


@profile('JSS20', report_printer=print)
def jss20(config: FL_HunterConfig):
    logger.add(config.output_dir / 'log')
    base = FDGModelInterface(config)
    cdp = base.fdg
    mp = base.metric_preprocessor
    train_ids = base.train_failure_ids
    test_ids = base.test_failure_ids
    # train_ids = [54, 28, 77]
    # test_ids = [69, 67, 6]
    graph_library = GraphLibrary(cdp, train_ids[:], mp)
    labels = []
    preds = []
    for fid, (_, fault) in zip(test_ids, cdp.failures_df.iloc[test_ids[:]].iterrows()):
        labels.append({fault['root_cause_node']})
        preds.append(graph_library.query(fid))
    metrics = get_evaluation_metrics_dict(labels, preds, max_rank=cdp.n_failure_instances)
    logger.info(metrics)
    return metrics


def main(config: FL_HunterConfig):
    metrics = jss20(config)
    profiler = Profiler().get('/JSS20')
    logger.info(format_result_string(metrics, profiler, config))


if __name__ == '__main__':
    main(FL_HunterConfig().parse_args())
