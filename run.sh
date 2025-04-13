#!/bin/bash

docker-compose -f docker-compose-3.yml up -d --build

sleep 5

docker exec namenode hdfs dfsadmin -safemode leave
docker exec namenode hdfs dfs -rm /data/data.csv
docker cp data.csv namenode:/

docker exec namenode hdfs dfs -mkdir -p /data
docker exec namenode hdfs dfs -D dfs.block.size=32M -put /data.csv /data/
docker exec namenode hdfs dfsadmin -setSpaceQuota 2g /data

docker cp app.py spark-master:/

docker exec spark-master apk add --update make automake gcc g++ python-dev linux-headers freetype-dev libpng-dev
docker exec spark-master pip install numpy psutil requests 
docker exec spark-master pip install matplotlib
docker exec spark-master /spark/bin/spark-submit --master spark://spark-master:7077 /app.py
docker cp spark-master:/spark/results.log ./
docker cp spark-master:/spark/memory_usage.png ./

docker-compose -f docker-compose-3.yml down
