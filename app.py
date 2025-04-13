# coding: utf-8
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, unix_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time, psutil, os
import logging
import requests
import json
import matplotlib.pyplot as plt

memory_usage_over_time = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_spark_metrics(sc):
    metrics = {
        'executors_used': 0.0,
        'driver_used': 0.0,
        'gc_time': 0
    }

    ui_url = sc.uiWebUrl
    app_id = sc.applicationId

    executors_url = "{}/api/v1/applications/{}/executors".format(ui_url, app_id)
    response = requests.get(executors_url)

    if response.status_code == 200:
        executors_data = json.loads(response.text)
        for executor in executors_data:
            if executor['id'] != 'driver':
                metrics['executors_used'] += executor['memoryUsed'] / (1024 ** 2) 

    runtime = sc._jvm.java.lang.management.ManagementFactory.getRuntimeMXBean()
    memory_bean = sc._jvm.java.lang.management.ManagementFactory.getMemoryMXBean()

    heap_usage = memory_bean.getHeapMemoryUsage()
    non_heap_usage = memory_bean.getNonHeapMemoryUsage()

    metrics['driver_used'] = (heap_usage.getUsed() + non_heap_usage.getUsed()) / (1024 ** 2)

    gc_beans = sc._jvm.java.lang.management.ManagementFactory.getGarbageCollectorMXBeans()
    metrics['gc_time'] = sum(gc.getCollectionTime() for gc in gc_beans)

    return metrics


def record_memory(experiment_name, metrics):
    if experiment_name not in memory_usage_over_time:
        memory_usage_over_time[experiment_name] = {
            'time': [],
            'driver_used': [],
            'executors_used': [],
            'total_used': []
        }
    memory_usage_over_time[experiment_name]['time'].append(time.time() - experiment_start_time)
    memory_usage_over_time[experiment_name]['driver_used'].append(metrics['driver_used'])
    memory_usage_over_time[experiment_name]['executors_used'].append(metrics['executors_used'])
    memory_usage_over_time[experiment_name]['total_used'].append(metrics['driver_used'] + metrics['executors_used'])


def plot_memory_usage():
    num_experiments = len(memory_usage_over_time)
    fig, axes = plt.subplots(nrows=num_experiments, ncols=1, figsize=(12, 6 * num_experiments))
    axes = axes.flatten() if num_experiments > 1 else [axes]

    for i, (experiment, data) in enumerate(memory_usage_over_time.items()):
        axes[i].plot(data['time'], data['driver_used'], label='Driver Memory (MB)')
        axes[i].plot(data['time'], data['executors_used'], label='Executors Memory (MB)')
        axes[i].plot(data['time'], data['total_used'], label='Total Memory (MB)')
        axes[i].set_xlabel("Time (seconds)")
        axes[i].set_ylabel("Memory Usage (MB)")
        axes[i].set_title("Memory Usage Over Time - {}".format(experiment))
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig('/spark/memory_usage.png')
    logger.info("Plots saved to /spark/memory_usage.png")


def run_experiment(data_path, experiment_name, cache_data=False, repartition_num=None, num_nodes=1):
    global experiment_start_time

    if num_nodes == 1:
        spark = SparkSession.builder.appName("BeverageSalesClassifier").master("spark://spark-master:7077").config("spark.driver.memory", "2g").config("spark.hadoop.dfs.blocksize", "128m").config("spark.logConf", "false").config("spark.eventLog.enabled", "false").getOrCreate()
    else:
        spark = SparkSession.builder.appName("BeverageSalesClassifier").master("spark://spark-master:7077").config("spark.executor.memory", "4g").config("spark.driver.memory", "1g").config("spark.hadoop.dfs.blocksize", "128m").config("spark.executor.instances", "3").config("spark.logConf", "false").config("spark.eventLog.enabled", "false").getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    experiment_start_time = time.time()
    logger.info("Experiment: {} - Starting...".format(experiment_name))
    initial_metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, initial_metrics)

    logger.info("Experiment: {} - Loading data from {}".format(experiment_name, data_path))
    data = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.info("Experiment: {} - Data loaded. Row count: {}, Columns: {}".format(experiment_name, data.count(), len(data.columns)))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    indexer = StringIndexer(inputCol="Customer_Type", outputCol="label")
    data = indexer.fit(data).transform(data)
    logger.info("Experiment: {} - Customer_Type indexed.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    data = data.withColumn("Order_Date", unix_timestamp("Order_Date", "yyyy-MM-dd").cast("double"))
    logger.info("Experiment: {} - Order_Date converted to timestamp.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    cat_cols = ["Product", "Category", "Region"]
    for col_name in cat_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
        data = indexer.fit(data).transform(data).drop(col_name)
    logger.info("Experiment: {} - Categorical features indexed.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    features = ["Order_Date", "Unit_Price", "Quantity", "Discount", "Total_Price"] + [c + "_index" for c in cat_cols]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data = assembler.transform(data)
    logger.info("Experiment: {} - Features assembled.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    if cache_data:
        data = data.cache()
        data.count()
        logger.info("Experiment: {} - Data cached.".format(experiment_name))
        metrics = get_spark_metrics(spark.sparkContext)
        record_memory(experiment_name, metrics)

    if repartition_num is not None:
        data = data.repartition(repartition_num)
        logger.info("Experiment: {} - Data repartitioned to {} partitions.".format(experiment_name, repartition_num))
        metrics = get_spark_metrics(spark.sparkContext)
        record_memory(experiment_name, metrics)

    train, test = data.randomSplit([0.7, 0.3], seed=42)
    logger.info("Experiment: {} - Data split into train and test sets.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train)
    logger.info("Experiment: {} - Logistic Regression model trained.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    predictions = model.transform(test)
    logger.info("Experiment: {} - Predictions generated.".format(experiment_name))
    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    metrics = get_spark_metrics(spark.sparkContext)
    record_memory(experiment_name, metrics)

    end_time = time.time()
    time_taken = end_time - experiment_start_time
    final_metrics = get_spark_metrics(spark.sparkContext)

    with open('/spark/results.log', 'a') as f:
        f.write("Experiment: {}\n".format(experiment_name))
        f.write("  Time: {:.2f} seconds\n".format(time_taken))
        f.write("  Driver Memory Used: {:.2f} MB\n".format(final_metrics['driver_used']))
        f.write("  Executors Memory Used: {:.2f} MB\n".format(final_metrics['executors_used']))
        f.write("  Total Memory Used: {:.2f} MB\n".format(final_metrics['driver_used'] + final_metrics['executors_used']))
        f.write("-" * 30 + "\n")

    if cache_data:
        data.unpersist()

    spark.stop()


if __name__ == "__main__":
    hdfs_path = "hdfs://namenode:9000/data/data.csv"

    # 1 DataNode, Spark
    logger.info("Starting Experiment 1: 1 DataNode, Spark")
    run_experiment(hdfs_path, "1_DataNode_Spark")

    # 1 DataNode, Spark Opt
    logger.info("Starting Experiment 2: 1 DataNode, Spark Optimized")
    run_experiment(hdfs_path, "1_DataNode_Spark_Opt", cache_data=True, repartition_num=8)

    # 3 DataNode, Spark
    logger.info("Starting Experiment 3: 3 DataNode, Spark")
    run_experiment(hdfs_path, "3_DataNode_Spark", num_nodes=3)

    # 3 DataNode, Spark Opt
    logger.info("Starting Experiment 4: 3 DataNode, Spark Optimized")
    run_experiment(hdfs_path, "3_DataNode_Spark_Opt", cache_data=True, repartition_num=8, num_nodes=3)

    plot_memory_usage()
