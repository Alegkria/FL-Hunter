import logging
import math

import numpy as np
import sklearn
from lime import explanation
from lime import lime_base

__all__ = ["LimeTimeSeriesExplainer"]

class TSDomainMapper(explanation.DomainMapper):

    def __init__(self, signal_names, num_slices, is_multivariate):
        """Init function.
        Args:
            signal_names: list of strings(一个字符串列表), names of signals(时间序列数据中的信号名称)
            num_slices: 整数值，表示时间序列数据在时间轴上的切片数量
            is_multivariate: 布尔值，表示时间序列数据是否为多变量（Multivariate）数据，即包含多个信号
        """
        super().__init__()
        self.num_slices = num_slices
        self.signal_names = signal_names
        self.is_multivariate = is_multivariate

    def map_exp_ids(self, exp, **kwargs):
        # in case of univariate, don't change feature ids
        if not self.is_multivariate:
            return exp

        names = []
        for _id, weight in exp:
            # from feature idx, extract both the pair number of slice
            # and the signal perturbed
            nsignal = int(_id / self.num_slices)
            nslice = _id % self.num_slices
            signalname = self.signal_names[nsignal]
            featurename = "%d - %s" % (nslice, signalname)
            names.append((featurename, weight))
        return names

class LimeTimeSeriesExplainer(object):
    """Explains time series classifiers."""


    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 signal_names=["not specified"]
                 ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel
             浮点数，表示指数核函数的宽度
            verbose: if true, print local prediction values from linear model
            布尔值，如果为True，则打印线性模型的局部预测值
            class_names: list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
                '1', ...
            一个字符串列表，表示分类器预测结果的类别名称，按照分类器的输出顺序排列。如果没有提供，类名将默认为 '0'、'1'、...。
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
            字符串，表示特征选择方法。可以是 'forward_selection'、'lasso_path'、'none' 或 'auto'。
            signal_names: list of strings, names of signals
            一个字符串列表，表示时间序列数据中的信号名称
        """

        # exponential kernel
        def kernel(d): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.base = lime_base.LimeBase(kernel, verbose)  # 创建实例
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.signal_names = signal_names

    def explain_instance(self,
                         timeseries_instance,
                         classifier_fn,
                         num_slices,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         model_regressor=None,
                         replacement_method='mean'):
        """Generates explanations for a prediction.
           用于生成对分类结果的解释

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).
        As distance function DTW metric is used.

        Args:
            time_series_instance: time series to be explained.分类器预测概率函数
            classifier_fn: classifier prediction probability function,
                which takes a list of d arrays with time series values
                and outputs a (d, k) numpy array with prediction
                probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
                分类器预测概率函数，它接受一个包含时间序列值的列表，
                并输出一个形状为 (d, k) 的numpy数组，
                其中k是类别的数量。
                对于Scikit-learn分类器，
                这通常是 classifier.predict_proba 函数。
            num_slices: Defines into how many slices the time series will
                be split up   定义时间序列将被分成多少个切片
            labels: iterable with labels to be explained. 一个可迭代对象，表示要解释的标签
            top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
            如果不为None，则忽略labels参数，并为概率最高的K个标签生成解释，其中K由这个参数确定。
            num_features: maximum number of features present in explanation
            解释中最多包含的特征数量
            num_samples: size of the neighborhood to learn the linear model 用于学习线性模型的邻域数据的大小
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
                sklearn回归器用于解释。默认为LimeBase中的Ridge回归器 必须具有 model_regressor.coef_ 和 sample_weight 作为参数来进行模型训练
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
       """

        permutations, predictions, distances = self.__data_labels_distances(
            timeseries_instance, classifier_fn,
            num_samples, num_slices, replacement_method)


        is_multivariate = len(timeseries_instance.shape) > 1


        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]


        domain_mapper = TSDomainMapper(self.signal_names, num_slices, is_multivariate)

        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)

        ret_exp.predict_proba = predictions[0]


        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[int(label)],
             ret_exp.local_exp[int(label)],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                permutations, predictions,
                distances, label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        return ret_exp

    def __data_labels_distances(cls,
                                timeseries,
                                classifier_fn,
                                num_samples,
                                num_slices,
                                replacement_method='mean'):
        """Generates a neighborhood around a prediction.
           私有静态方法 用于生成预测结果的邻域数据和距离信息，
           以支持LIME（Local Interpretable Model-agnostic Explanations）对时间序列分类器的解释

        Generates neighborhood data by randomly removing slices from the
        time series and replacing them with other data points (specified by
        replacement_method: mean over slice range, mean of entire series or
        random noise). Then predicts with the classifier.

        Args:
            timeseries: Time Series to be explained.要解释的时间序列数据
                it can be a flat array (univariate)
                or (num_signals, num_points) (multivariate)
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
                分类器的预测概率函数，它接受一个时间序列，并输出预测概率。对于Scikit-learn分类器，通常是 classifier.predict_proba 函数
            num_samples: size of the neighborhood to learn the linear 邻域数据的大小，用于学习局部加权线性模型
                model (perturbation + original time series)
            num_slices: how many slices the time series will be split into
                for discretization.  将时间序列分成多少个切片
            replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
                定义如何将时间序列中的切片停用（替换）为其他数据点 可选值为 'mean'、'total_mean' 和 'noise'
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of slices in the time series. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: distance between the original instance and
                    each perturbed instance
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='cosine').ravel() * 100

        num_channels = 1
        len_ts = len(timeseries)
        if len(timeseries.shape) > 1:
            num_channels, len_ts = timeseries.shape

        values_per_slice = math.ceil(len_ts / num_slices)
        # 计算每个切片中的值的数量，用于划分时间序列
        deact_per_sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_channels, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries.copy()]

        for i, num_inactive in enumerate(deact_per_sample, start=1):

            logging.info("sample %d, inactivating %d", i, num_inactive)
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(features_range, num_inactive,
                                             replace=False)
            num_channels_to_perturb = np.random.randint(1, num_channels + 1)


            channels_to_perturb = np.random.choice(range(num_channels),
                                                   num_channels_to_perturb,
                                                   replace=False)

            logging.info("sample %d, perturbing signals %r", i,
                         channels_to_perturb)

            for chan in channels_to_perturb:
                perturbation_matrix[i, chan, inactive_idxs] = 0

            tmp_series = timeseries.copy()


            for idx in inactive_idxs:
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len_ts)


                if replacement_method == 'mean':
                    # use mean of slice as inactive
                    perturb_mean(tmp_series, start_idx, end_idx,
                                 channels_to_perturb)
                elif replacement_method == 'noise':
                    # use random noise as inactive
                    perturb_noise(tmp_series, start_idx, end_idx,
                                  channels_to_perturb)
                elif replacement_method == 'total_mean':
                    # use total series mean as inactive
                    perturb_total_mean(tmp_series, start_idx, end_idx,
                                       channels_to_perturb)
            original_data.append(tmp_series)

        predictions = classifier_fn(np.array(original_data))

        # create a flat representation for features
        perturbation_matrix = perturbation_matrix.reshape((num_samples, num_channels * num_slices))
        distances = distance_fn(perturbation_matrix)


        return perturbation_matrix, predictions, distances

def perturb_total_mean(m, start_idx, end_idx, channels):

    if len(m.shape) == 1:
        m[start_idx:end_idx] = m.mean()
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = m[chan].mean()


def perturb_mean(m, start_idx, end_idx, channels):

    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.mean(m[start_idx:end_idx])
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.mean(m[chan][start_idx:end_idx])


def perturb_noise(m, start_idx, end_idx, channels):

    if len(m.shape) == 1:
        m[start_idx:end_idx] = np.random.uniform(m.min(), m.max(),
                                                 end_idx - start_idx)
        return

    for chan in channels:
        m[chan][start_idx:end_idx] = np.random.uniform(m[chan].min(),
                                                       m[chan].max(),
                                                       end_idx - start_idx)
