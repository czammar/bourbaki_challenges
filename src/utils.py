from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import keyword_only, since, SparkContext, inheritable_thread_target

from pyspark.tuning import (
    CrossValidatorModel,
    _CrossValidatorParams,
    _parallelFitTasks
)

from pyspark.ml.param.shared import HasCollectSubModels, HasParallelism
from pyspark.ml import Estimator, Model
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.util import (
    MLReadable,
    MLWritable
)

from pyspark.ml.param import Params, Param, TypeConverters

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
    TYPE_CHECKING,
)

from multiprocessing.pool import ThreadPool
import numpy as np

class TimeSeriesCrossValidator(
    Estimator["CrossValidatorModel"],
    _CrossValidatorParams,
    HasParallelism,
    HasCollectSubModels,
    MLReadable["CrossValidator"],
    MLWritable
    ):
    """
    WIP
    """

    dateCol = Param(Params._dummy(), "dateCol", "Name of column that holds the date of reference for Cross Validation",
                      typeConverter=TypeConverters.toString)

    blockingTime = Param(Params._dummy(), "False", "Define is time series cross validation takes a block of chunck size or all the past",
                      typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self,
                 estimator=None,
                 estimatorParamMaps=None,
                 evaluator=None,
                 seed=None,
                 dateCol='dateCol',
                 blockingTime=False,
                 parallelism=1):

        super(TimeSeriesCrossValidator, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @staticmethod
    def _gen_avg_and_std_metrics(metrics_all: List[List[float]]) -> Tuple[List[float], List[float]]:
        avg_metrics = np.mean(metrics_all, axis=0)
        std_metrics = np.std(metrics_all, axis=0)
        return list(avg_metrics), list(std_metrics)

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        dateCol = self.getOrDefault(self.dateCol)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        n = dataset.count()
        chunk_size = int(n/(nFolds+1))
        metrics_all = [[0.0] * numModels for i in range(nFolds)]

        seed = self.getOrDefault(self.seed)

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]
    
        dataset = dataset.withColumn(
            "row_id",
            F.row_number().over(
                Window.partitionBy().orderBy(dateCol)
                )
                )

        datasets = self._kFold(dataset)

        for i in range(nFolds):
            validation = datasets[i][1].cache()
            train = datasets[i][0].cache()

            tasks = map(
                inheritable_thread_target,
                _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics_all[i][j] = metric
                if collectSubModelsParam:
                    assert subModels is not None
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        metrics, std_metrics = TimeSeriesCrossValidator.\
            _gen_avg_and_std_metrics(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])

        return self._copyValues(
            CrossValidatorModel(bestModel, metrics, cast(List[List[Model]], subModels), std_metrics)
        )


    def _kFold(self, dataset: DataFrame) -> List[Tuple[DataFrame, DataFrame]]:
        nFolds = self.getOrDefault(self.numFolds)
        blockingTime = self.getOrDefault(self.blockingTime)
        n = dataset.count()
        chunk_size = int(n/(nFolds+1))

        datasets = []

        # Use user-specified fold numbers.
        for i in range(nFolds):

            if blockingTime is False:
                training = dataset.filter(
                    F.col('row_id') <= chunk_size * (i+1)
                    ).cache()

            else:
                training = dataset.filter(
                    (F.col('row_id') > chunk_size * i) &
                    (F.col('row_id') <= chunk_size * (i+1))
                    ).cache()

        validation = dataset.filter(
                    (F.col('row_id') > chunk_size * (i+1)) &
                    (F.col('row_id') <= chunk_size * (i+2))
                    ).cache()

        if training.rdd.getNumPartitions() == 0 or len(training.take(1)) == 0:
            raise ValueError("The training data at fold %s is empty." % i)

        if validation.rdd.getNumPartitions() == 0 or len(validation.take(1)) == 0:
            raise ValueError("The validation data at fold %s is empty." % i)

        datasets.append((training, validation))

        return datasets
