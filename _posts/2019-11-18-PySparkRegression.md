---
title: " Predict Trip duration of a Passenger using Pyspark"
date: 2019-11-18
tags: [PySpark,Regression]
---
# This Project implements ensemble regression tree and highlights how it is different from regression tree 

```python
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf,col
from datetime import datetime
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
sc= SparkContext()
sqlContext = SQLContext(sc)

train = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('C:/train.csv')
#train.show()
```
reference: stackoverflow.com

# Feature Engineering
Here we are using lat-lon data to create a new feature for trip distance using Haversine formula 

```python
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6372.8 # Radius of earth in kilometers. Use 3956 for miles
    distance =  c * r
    return abs(round(distance, 2))
```
Spark stores data in dataframes or RDDs. Here I have used dataframes hence I cannot create custom functions without registering the funstion first i.e to save it as if it were one of the built-in database functions first before 
using it. Thats where  User Defined Functions (UDF) comes in.

```python
udf_get_distance = F.udf(haversine)
```

```python
train = (train.withColumn('DISTANCE', udf_get_distance(
train.pickup_longitude, train.pickup_latitude,
train.dropoff_longitude, train.dropoff_latitude)))
columns_to_drop = ['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude']
train = train.drop(*columns_to_drop)
train.describe().show()
```

    +-------+---------+------------------+------------------+------------------+-----------------+------------------+
    |summary|       id|         vendor_id|   passenger_count|store_and_fwd_flag|    trip_duration|          DISTANCE|
    +-------+---------+------------------+------------------+------------------+-----------------+------------------+
    |  count|  1458644|           1458644|           1458644|           1458644|          1458644|           1458644|
    |   mean|     null|1.5349502688798637|1.6645295219395548|              null|959.4922729603659|2.1365657007467114|
    | stddev|     null|0.4987771539074011|1.3142421678231184|              null|5237.431724497624| 2.667889495100226|
    |    min|id0000001|                 1|                 0|                 N|                1|               0.0|
    |    max|id4000000|                 2|                 9|                 Y|          3526282|              9.99|
    +-------+---------+------------------+------------------+------------------+-----------------+------------------+


```python
train = train.withColumn('store_and_fwd_flag', F.when(train.store_and_fwd_flag == 'N', 0).otherwise(1))
train.take(1)
```

    [Row(id='id2875421', vendor_id=2, pickup_datetime=datetime.datetime(2016, 3, 14, 17, 24, 55), dropoff_datetime=datetime.datetime(2016, 3, 14, 17, 32, 30), passenger_count=1, store_and_fwd_flag=0, trip_duration=455, DISTANCE='1.5')]

```python
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

split_col = pyspark.sql.functions.split(train['pickup_datetime'], ' ')
train = train.withColumn('Date', split_col.getItem(0))
train = train.withColumn('Time', split_col.getItem(1))
split_date=pyspark.sql.functions.split(train['Date'], '-')     
train= train.withColumn('Year', split_date.getItem(0))
train= train.withColumn('Month', split_date.getItem(1))
train= train.withColumn('Day', split_date.getItem(2))
funcWeekDay =  udf(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%w'))
train = train.withColumn('weekDay', funcWeekDay(col('Date')))
split_date=pyspark.sql.functions.split(train['Time'], ':')     
train= train.withColumn('hour', split_date.getItem(0))
train= train.withColumn('minutes', split_date.getItem(1))
train= train.withColumn('seconds', split_date.getItem(2))
train.take(1)

```

    [Row(id='id2875421', vendor_id=2, pickup_datetime=datetime.datetime(2016, 3, 14, 17, 24, 55), dropoff_datetime=datetime.datetime(2016, 3, 14, 17, 32, 30), passenger_count=1, store_and_fwd_flag=0, trip_duration=455, DISTANCE='1.5', Date='2016-03-14', Time='17:24:55', Year='2016', Month='03', Day='14', weekDay='1', hour='17', minutes='24', seconds='55')]


```python
columns_to_drop = ['pickup_datetime','dropoff_datetime', 'Date','Time']
train = train.drop(*columns_to_drop)
```

    +---------+---------+---------------+------------------+-------------+--------+----+-----+---+-------+----+-------+-------+
    |       id|vendor_id|passenger_count|store_and_fwd_flag|trip_duration|DISTANCE|Year|Month|Day|weekDay|hour|minutes|seconds|
    +---------+---------+---------------+------------------+-------------+--------+----+-----+---+-------+----+-------+-------+
    |id2875421|        2|              1|                 0|          455|       0|2016|    3| 14|      1|  17|     24|     55|
    +---------+---------+---------------+------------------+-------------+--------+----+-----+---+-------+----+-------+-------+
    only showing top 1 row


```python
from pyspark.sql.functions import log
train = train.withColumn("log_duration", log(train["trip_duration"]) )
train.show(1)
```

    +---------+---------+---------------+------------------+-------------+--------+----+-----+---+-------+----+-------+-------+----------------+
    |       id|vendor_id|passenger_count|store_and_fwd_flag|trip_duration|DISTANCE|Year|Month|Day|weekDay|hour|minutes|seconds|    log_duration|
    +---------+---------+---------------+------------------+-------------+--------+----+-----+---+-------+----+-------+-------+----------------+
    |id2875421|        2|              1|                 0|          455|       1|2016|    3| 14|      1|  17|     24|     55|6.12029741895095|
    +---------+---------+---------------+------------------+-------------+--------+----+-----+---+-------+----+-------+-------+----------------+
    only showing top 1 row


```python
train.cache()
train.printSchema()
```

    root
     |-- id: string (nullable = true)
     |-- vendor_id: integer (nullable = true)
     |-- passenger_count: integer (nullable = true)
     |-- store_and_fwd_flag: integer (nullable = false)
     |-- trip_duration: integer (nullable = true)
     |-- DISTANCE: integer (nullable = true)
     |-- Year: integer (nullable = true)
     |-- Month: integer (nullable = true)
     |-- Day: integer (nullable = true)
     |-- weekDay: integer (nullable = true)
     |-- hour: integer (nullable = true)
     |-- minutes: integer (nullable = true)
     |-- seconds: integer (nullable = true)
     |-- log_duration: double (nullable = true)


```python
train.describe().toPandas()

```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>id</th>
      <th>vendor_id</th>
      <th>passenger_count</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
      <th>DISTANCE</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>weekDay</th>
      <th>hour</th>
      <th>minutes</th>
      <th>seconds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
      <td>1458644</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>None</td>
      <td>1.5349502688798637</td>
      <td>1.6645295219395548</td>
      <td>1.0</td>
      <td>959.4922729603659</td>
      <td>3.4408639325291177</td>
      <td>2016.0</td>
      <td>3.516817674497684</td>
      <td>15.504018115455176</td>
      <td>3.1128177951576945</td>
      <td>13.60648451575573</td>
      <td>29.59015770811795</td>
      <td>29.473590540255195</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>None</td>
      <td>0.4987771539074011</td>
      <td>1.3142421678231184</td>
      <td>0.0</td>
      <td>5237.431724497624</td>
      <td>4.296542880941734</td>
      <td>0.0</td>
      <td>1.6810375087348843</td>
      <td>8.703135115281617</td>
      <td>1.9928049699468888</td>
      <td>6.399692034352387</td>
      <td>17.324714120895614</td>
      <td>17.319851679258015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>id0000001</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>id4000000</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>3526282</td>
      <td>97.59</td>
      <td>2016</td>
      <td>6</td>
      <td>31</td>
      <td>6</td>
      <td>23</td>
      <td>59</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>


```python
vectorAssembler = VectorAssembler(inputCols = ['vendor_id', 'passenger_count', 'store_and_fwd_flag','DISTANCE', 'Year', 'Month','Day','weekDay','hour','minutes','seconds'], outputCol = 'features')
vtrain = vectorAssembler.transform(train)
vtrain = vtrain.select(['features', 'log_duration'])
vtrain.show(3)
```

    +--------------------+------------------+
    |            features|      log_duration|
    +--------------------+------------------+
    |[2.0,1.0,0.0,1.0,...|  6.12029741895095|
    |[1.0,1.0,0.0,1.0,...|6.4967749901858625|
    |[2.0,1.0,0.0,6.0,...|  7.66105638236183|
    +--------------------+------------------+
    only showing top 3 rows


```python
splits = vtrain.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'log_duration',maxDepth = 2)
#dt_model = dt.fit(train_df)
#dt_predictions = dt_model.transform(train_df)
#DRt
dt_evaluator = RegressionEvaluator(labelCol="log_duration", predictionCol="prediction", metricName="rmse")
#rmse = dt_evaluator.evaluate(dt_predictions)
#print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
```
# Parameter tuning in Pyspark

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

dtparamGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [2, 3,5])
             .addGrid(dt.maxBins, [4, 8,32])
             .build())

dtcv = CrossValidator(estimator = dt,
                      estimatorParamMaps = dtparamGrid,
                      evaluator = dt_evaluator,
                      numFolds = 10)

# Run cross validations
dtcvModel = dtcv.fit(train_df)
print(dtcvModel)
dtpredictions = dtcvModel.transform(test_df)
mae = dt_evaluator.evaluate(dtpredictions)
print(" Mean Absolute Error (MAE) on test data = %g" % mae)
```

     Mean Absolute Error (MAE) on test data = 0.340612


```python
rmse = dt_evaluator.evaluate(dtpredictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
```

    Root Mean Squared Error (RMSE) on test data = 0.513194


```python
dtpredictions_ert = dtcvModel.transform(train_df)

```


```python
dtpredictions_ert.show(2)
```


    [Row(features=SparseVector(11, {0: 1.0, 1: 2.0, 4: 2016.0, 5: 2.0, 6: 14.0, 9: 17.0}), log_duration=5.3230099791384085, prediction=5.341284741186751),
     Row(features=SparseVector(11, {0: 2.0, 1: 1.0, 4: 2016.0, 5: 5.0, 6: 15.0, 9: 15.0}), log_duration=5.814130531825066, prediction=5.341284741186751)]


```python
dtpredictions.show(2)
```


    [Row(features=SparseVector(11, {0: 2.0, 1: 1.0, 4: 2016.0, 5: 2.0, 6: 28.0, 9: 23.0}), log_duration=5.493061443340548, prediction=5.711890426639655),
     Row(features=SparseVector(11, {0: 2.0, 1: 2.0, 4: 2016.0, 5: 2.0, 6: 7.0, 9: 55.0}), log_duration=5.54907608489522, prediction=5.711890426639655)]


```python
pred_table = dtpredictions.groupBy('prediction').count()
pred_table = pred_table.drop('count')
pred_table.show()
```

    +------------------+
    |        prediction|
    +------------------+
    | 6.284032803395688|
    |6.0485785145505995|
    | 5.946427673262689|
    | 6.141958153601273|
    | 6.551517246541201|
    |7.1425228143753525|
    | 7.545394329958232|
    | 7.298411167160363|
    |5.7496543843581795|
    | 7.924025245291648|
    |6.7242448681574745|
    +------------------+
    only showing top 20 rows




```python
pred_table1 = dtpredictions_ert.groupBy('prediction').count()
pred_table1.show()
predlist = pred_table.select("prediction").collect()
predlist = pred_table.select("prediction").rdd.flatMap(lambda x: x).collect()
predlist[0]
```
    6.284032803395688


```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

lr = LinearRegression(featuresCol ='features', labelCol = 'log_duration',maxIter=10)
#lr_model = lr.fit(newdf)
evaluator = RegressionEvaluator(labelCol="log_duration", predictionCol="prediction", metricName="rmse")


paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01,0.3,0.5,1]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.1,0.2, 0.5, 0.8,1.0])\
    .build()

tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8)


```

```python
for i in predlist:
    newdf = dtpredictions.filter(dtpredictions.prediction == i)
    newdft = dtpredictions_ert.filter(dtpredictions_ert.prediction == i)
    newdf = newdf.drop('prediction')
    newdft = newdft.drop('prediction')
    model = tvs.fit(newdft)
    lr_predictions = model.transform(newdf)
    rmse = dt_evaluator.evaluate(lr_predictions)
    print("mae on test data = %g" % mae)

```
Average MAE = 0.32
    
