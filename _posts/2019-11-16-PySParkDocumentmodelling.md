---
title: "PySpark: Document Classification "
date: 2019-11-16
tags: [PySpark, NLP]
---

# Text Classification

# PySpark DataFrames

PySpark DataFrames is an implementation of the pairs method in Spark, using frames.

# About Data
Here For demonstration of Document modelling in PySpark we are using State of the Union (SOTU) texts which provides access to the corpus of all the State of the Union addresses from 1790 to 2019.
*SOTU* maps the significant content of each State of the Union address so that users can appreciate its key terms and their relative importance.
The current corpus contains 233 documents. There are **1,785,586** words in the corpus, and **28,072** unique words.

```python
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)
```

```python
from itertools import combinations
from pyspark.ml.feature import NGram
from pyspark.sql import Row
from pyspark.ml.feature import StopWordsRemover
import numpy as np
```
Document lines are transformed as rdd and then converted it into **DataFrame**

```python
def fun(lines):
    lines = lines.split()
    return lines

lines = sc.textFile("C:\\Users\\dpawa\\OneDrive\\Documents\\stateoftheunion1790-2019.txt").map(fun)

```

In order to understand trending words from the speech We need to filter out stop words.
```python
row = Row('sentence') # Or some other column name
df = lines.map(row).toDF()
remover = StopWordsRemover(inputCol="sentence", outputCol="filtered")
filter_list = remover.transform(df)
filter_list = filter_list.drop('sentence')
rdd1 = filter_list.rdd.flatMap(lambda x: x)
```
List has to be converted into set here as we need combinations of all words but we need a non-redundant list.
**NOTE** : Set is a unordered immutable data structure in python which is helpful to prevent redundancy in a list when needed.
```python
rdd_combo = rdd1.flatMap(lambda x: list(set(combinations(x,2))))
rdd_combo.take(1)
```

[('george', 'favorable')]


```python
rdd_f = rdd_combo.map(lambda x : (x , 1)).reduceByKey(lambda a, b: a+b).filter(lambda x : x[1] > 5)

```

```python
onedf = rdd_f.map(lambda x : (x[0][0],x[1]))
table1 = spark.createDataFrame(onedf,['onedf','count_f'])
twodf = rdd_f.map(lambda x : (x[0][1],x[1]))
table2 = spark.createDataFrame(twodf,['twodf','count_s'])
```

```python
first = rdd_f.map(lambda x : (x[0][0],x[0][1],x[1]))
table = spark.createDataFrame(first,['first','second','count1'])
```

```python
import pyspark.sql.functions as f

table3 = filter_list.withColumn('word', f.explode(f.col('filtered')))\
    .groupBy('word')\
    .count()

table3.show()
```
    +-------------+-----+
    |         word|count|
    +-------------+-----+
    |      embrace|   47|
    |         hope|  842|
    |        still| 1055|
    |  transaction|   44|
    |    standards|  166|
    |apprehensions|   37|
    |    connected|  177|
    |      implore|    4|
    |gratification|   41|
    +-------------+-----+

```python
import pyspark.sql.functions as F

ta = table.alias('ta')
tb = table3.alias('tb')


inner_join = table.join(tb, ta.second == tb.word)\
    .withColumn("P(A/B)", (F.col("count1") / F.col("count")))
inner_join.show()
```
In PySpark we can use SQL functions in order to manipulate data and get desired transformations.

    +-------------+------------+------+------------+-----+--------------------+
    |        first|      second|count1|        word|count|              P(A/B)|
    +-------------+------------+------+------------+-----+--------------------+
    |      prevent|accumulation|     8|accumulation|   50|                0.16|
    |        great|accumulation|     6|accumulation|   50|                0.12|
    |          may|accumulation|     8|accumulation|   50|                0.16|
    |   government|accumulation|     6|accumulation|   50|                0.12|
    |        given|  commanders|     6|  commanders|   58| 0.10344827586206896|
    |         last|  commanders|     7|  commanders|   58|  0.1206896551724138|
    |     military|  commanders|    16|  commanders|   58| 0.27586206896551724|
    |        naval|  commanders|    15|  commanders|   58| 0.25862068965517243|
    |         many|   connected|     9|   connected|  177| 0.05084745762711865|
    +-------------+------------+------+------------+-----+--------------------+


```python
from pyspark.sql.functions import col
inner_join.sort(col("count1").desc()).show()
```
For Example: Here We are finding probability(First_word_list/Second_word_list) which shows us the words which have more probability of co-existing together.

    +----------+----------+------+----------+-----+-------------------+
    |     first|    second|count1|      word|count|             P(A/B)|
    +----------+----------+------+----------+-----+-------------------+
    |    united|    states|  4132|    states| 6508| 0.6349108789182545|
    |government|    states|   729|    states| 6508| 0.1120159803318992|
    |    fiscal|      year|   705|      year| 3946|0.17866193613786113|
    |    states|    states|   634|    states| 6508|0.09741856177012907|
    |      last|      year|   626|      year| 3946|0.15864166244298022|
    |    states|government|   570|government| 7032| 0.0810580204778157|
    |government|    united|   563|    united| 4847|0.11615432226119249|
    +----------+----------+------+----------+-----+-------------------+
