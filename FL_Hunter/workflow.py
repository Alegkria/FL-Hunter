import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, List, Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch as th
from diskcache import Cache
from loguru import logger
from pyprof import profile, Profiler
from torchinfo import summary

from FL_Hunter.config import FL_HunterConfig
from FL_Hunter.dataset import prepare_sklearn_dataset
from FL_Hunter.evaluation_metrics import top_1_accuracy, top_2_accuracy, top_3_accuracy, top_k_accuracy, MAR
from FL_Hunter.models.get_model import ClassifierProtocol  # -------------------------------------------------
from FL_Hunter.models.interface import FL_HunterModelInterface, FL_HunterModuleProtocol
from FL_Hunter.models.interface.callbacks import CFLLoggerCallback, TestCallback
from failure_dependency_graph import FDG, FDGBaseConfig
from utils import count_parameters, plot_module_with_dot
from utils.callbacks import EpochTimeCallback
from utils.load_model import best_checkpoint


@profile("train_exp_CFL", report_printer=lambda _: logger.info(f"Time Report:\n{_}"))
def _train_exp_CFL(
        config: FL_HunterConfig,
        get_model: Callable[[FDG, FL_HunterConfig], FL_HunterModuleProtocol],
        plot_model: bool = False,
):
    warnings.filterwarnings("ignore")

    logger.info(
        f"\n================================================Config=============================================\n"
        f"{config!s}"
        f"\n===================================================================================================\n"
    )
    logger.info(f"reproducibility info: {config.get_reproducibility_info()}")
    logger.add(config.output_dir / 'log')
    config.save(str(config.output_dir / "config"))


    model = FL_HunterModelInterface(config=config, get_model=get_model)

    if plot_model:
        try:
            plot_module_with_dot(model, output_dir=config.output_dir, init_data=model.example_input_array[0])
        except Exception as e:
            logger.error(f"Plot Model Error: {e!r}")
    try:
        logger.info(
            f"\n==========================Model Summary==============================================="
            f"{summary(model.module, input_data=model.example_input_array, verbose=0, depth=8)}"
            f"======================================================================================\n"
        )
    except Exception as e:
        count_parameters(model.module)
        logger.error(f"Model Summary Error: {e!r}")


    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint_metric, save_last=True, save_top_k=5, mode='min',
            # remember to update parse_log when changing the filename
            filename="{epoch}-{A@1:.6f}-{val_loss:.6f}-{MAR:.6f}",
        ),
        CFLLoggerCallback(epoch_freq=config.display_epoch_freq, second_freq=config.display_second_freq),
        TestCallback(
            epoch_freq=config.test_epoch_freq,
            second_freq=config.test_second_freq,
            model=model,  # 表示被测试的模型对象
        ),
        EpochTimeCallback(printer=logger.info),
    ]

    if config.es:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=config.checkpoint_metric,
                patience=config.early_stopping_epoch_patience // config.valid_epoch_freq,
                mode='min',
                strict=False,
                verbose=True,
            ),
        )

    trainer = pl.Trainer(
        auto_lr_find=True,
        max_epochs=config.max_epoch,
        callbacks=callbacks,
        default_root_dir=str(config.output_dir),
        check_val_every_n_epoch=config.valid_epoch_freq,
        num_sanity_val_steps=-1,
        enable_progress_bar=False,
        gradient_clip_val=config.gradient_clip_val,
        auto_select_gpus=True if config.cuda else False,
        gpus=1 if config.cuda else 0,
    )
    trainer.fit(model=model)
    logger.info(f"{trainer.checkpoint_callback.best_model_path=}")
    logger.info(config.get_reproducibility_info())

    # The length of the list corresponds to the number of test dataloaders used.
    metrics, *_ = trainer.test(
        model, ckpt_path=str(best_checkpoint(config.output_dir)), dataloaders=model.test_dataloader()
    )
    return {
        "trainer": trainer,
        "model": model,
        "metrics": metrics,
    }


def _pickle_callback(result: Dict, output_dir: Path):

     model = result["model"]
    th.save(model.module, output_dir / 'module.pt')



def train_exp_CFL(
        config: FL_HunterConfig, get_model: Callable[[FDG, FL_HunterConfig], FL_HunterModuleProtocol], plot_model: bool = False,
        after_callbacks: Optional[List[Callable[[Dict], Any]]] = None
):

    if after_callbacks is None:
        after_callbacks = [
            lambda _: logger.info(format_result_string(
                metrics=_["metrics"], profiler=Profiler.get("/train_exp_CFL"), config=config,
            )),
            partial(_pickle_callback, output_dir=config.output_dir)
        ]
    result = _train_exp_CFL(config, get_model, plot_model=plot_model)

    for callback in after_callbacks:
        try:
            callback(result)
        except Exception as e:
            logger.error(f"Run callback error: {e!r}")
    print(f"train finished. saved to {config.output_dir}")


@profile(
    "train_exp_sklearn_classifier", report_printer=lambda _: logger.info(f"Time Report:\n{_}")
)
def __train_exp_sklearn_classifier(config: FL_HunterConfig, get_model: Callable[[FDG, FL_HunterConfig], ClassifierProtocol]):

    logger.add(config.output_dir / 'log')
    cdp, real_paths = FDG.load(
        dir_path=config.data_dir,
        graph_path=config.graph_config_path, metrics_path=config.metrics_path, faults_path=config.faults_path,
        return_real_path=True
    )  # 数据加载和预处理
    dataset_cache_dir = config.cache_dir / ".".join(
        f"{k}={real_paths[k]}" for k in sorted(real_paths.keys())
    ).replace('/', '_')
    logger.info(f"dataset_cache_dir={dataset_cache_dir}")
    cache = Cache(str(dataset_cache_dir), size_limit=int(1e10))
    del dataset_cache_dir, real_paths  # 释放缓存

    dataset, (_, _, test_fault_ids) = prepare_sklearn_dataset(cdp, config, cache, mode=config.ts_feature_mode)


    y_probs = np.zeros((len(test_fault_ids), cdp.n_failure_instances), dtype=np.float32)
    for fault_id in test_fault_ids:
        y_trues.append(set(map(cdp.instance_to_gid, cdp.failures_df.iloc[fault_id]['root_cause_node'].split(';'))))
    models = {}
    for node_type in cdp.failure_classes:  # 遍历 cdp.failure_classes 列表中的每个故障类别
        feature_names, (
            (train_x, train_y, _, _), _, (test_x, _, fault_ids, node_names)
        ) = dataset[node_type]  # 从 dataset 中提取了与当前故障类别 node_type 相关的数据


        if len(np.unique(train_y)) > 1:
            model = get_model(cdp, config)
            with profile("Training"):
                model.fit(train_x, train_y)

            models[node_type] = model
            with open(config.output_dir / f"{model.__class__}.{node_type=}.pkl", 'wb+') as f:
                pickle.dump(model, f)
            _y_probs = model.predict_proba(test_x)

        else:
            _y_probs = np.zeros((len(test_x), 2), dtype=np.float32)
            _y_probs[:, 0] = 1.0
        for fault_id, node_name, prob in zip(fault_ids, node_names,
                                             _y_probs):
            with profile("Inference for each failure"):
                y_probs[test_fault_ids.index(fault_id), cdp.instance_to_gid(node_name)] = 1 - prob[0].item()
    y_preds = [np.arange(len(prob))[np.argsort(prob, axis=-1)[::-1]].tolist() for prob in y_probs]
    metrics = {
        "A@1": top_1_accuracy(y_trues, y_preds),
        "A@2": top_2_accuracy(y_trues, y_preds),
        "A@3": top_3_accuracy(y_trues, y_preds),
        "A@5": top_k_accuracy(y_trues, y_preds, k=5),
        "MAR": MAR(y_trues, y_preds, max_rank=cdp.n_failure_instances),
    }
    return metrics




def train_exp_sklearn_classifier(config: FL_HunterConfig, get_model: Callable[[FDG, FL_HunterConfig], ClassifierProtocol]):
    metrics = __train_exp_sklearn_classifier(config, get_model)
    profiler = Profiler.get('/train_exp_sklearn_classifier')
    logger.info(format_result_string(metrics, profiler, config))



def format_result_string(metrics: Dict[str, float], profiler: Profiler, config: FDGBaseConfig) -> str:
    return (
        f"command output one-line summary: "
        f"{metrics['A@1'] * 100:.2f},{metrics['A@2'] * 100:.2f},{metrics['A@3'] * 100:.2f},"
        f"{metrics['A@5'] * 100:.2f},{metrics['MAR']:.2f},"
        f"{profiler.total},{''},{''},{config.output_dir},{''},{''},{''},"
        f"{config.get_reproducibility_info()['command_line']},"
        f"{config.get_reproducibility_info().get('git_url', '')}"
    )
