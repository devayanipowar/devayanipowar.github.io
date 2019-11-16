

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

```


```python

```


```python
from itertools import combinations
from pyspark.ml.feature import NGram
from pyspark.sql import Row
from pyspark.ml.feature import StopWordsRemover
import numpy as np
```


```python
def fun(lines):
    lines = lines.split()
    return lines

lines = sc.textFile("C:\\Users\\dpawa\\OneDrive\\Documents\\stateoftheunion1790-2019.txt").map(fun)

```


```python

```


```python
row = Row('sentence') # Or some other column name
df = lines.map(row).toDF()
remover = StopWordsRemover(inputCol="sentence", outputCol="filtered")
filter_list = remover.transform(df)
filter_list = filter_list.drop('sentence')
rdd1 = filter_list.rdd.flatMap(lambda x: x)
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
    |          art|   51|
    | accumulation|   50|
    |     inimical|    7|
    |       spared|   39|
    |  transmitted|  126|
    |         clog|    2|
    |precautionary|   14|
    |    involving|   85|
    |    destitute|   27|
    |  unequivocal|   11|
    |  unavoidably|   12|
    |gratification|   41|
    +-------------+-----+
    only showing top 20 rows
    
    


```python
import pyspark.sql.functions as F

ta = table.alias('ta')
tb = table3.alias('tb')


inner_join = table.join(tb, ta.second == tb.word)\
    .withColumn("P(A/B)", (F.col("count1") / F.col("count")))
inner_join.show()
```

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
    |    secretary|   connected|    10|   connected|  177| 0.05649717514124294|
    |      service|   connected|    15|   connected|  177|  0.0847457627118644|
    |      subject|   connected|     9|   connected|  177| 0.05084745762711865|
    |circumstances|   connected|     7|   connected|  177| 0.03954802259887006|
    |   government|   connected|    11|   connected|  177|0.062146892655367235|
    |       report|   connected|    10|   connected|  177| 0.05649717514124294|
    |     question|   connected|     6|   connected|  177| 0.03389830508474576|
    |          men|   connected|     7|   connected|  177| 0.03954802259887006|
    |    important|   connected|     8|   connected|  177| 0.04519774011299435|
    |          war|   connected|    11|   connected|  177|0.062146892655367235|
    |         last|   connected|     7|   connected|  177| 0.03954802259887006|
    +-------------+------------+------+------------+-----+--------------------+
    only showing top 20 rows
    
    


```python
from pyspark.sql.functions import col
inner_join.sort(col("count1").desc()).show()
```

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
    |government|government|   549|government| 7032| 0.0780716723549488|
    |  congress|    states|   503|    states| 6508|0.07728948985863553|
    |     great|   britain|   496|   britain|  530| 0.9358490566037736|
    |government|    people|   474|    people| 4017| 0.1179985063480209|
    |     state|     union|   461|     union| 1330|0.34661654135338343|
    |    united|government|   448|government| 7032|0.06370875995449374|
    |    states|    united|   447|    united| 4847|0.09222199298535176|
    |   federal|government|   445|government| 7032|0.06328213879408419|
    |  american|    people|   442|    people| 4017|0.11003236245954692|
    |      last|   session|   418|   session|  805| 0.5192546583850932|
    |      upon|government|   391|government| 7032|0.05560295790671217|
    |      last|  congress|   388|  congress| 5025| 0.0772139303482587|
    |    united|    united|   379|    united| 4847| 0.0781926965133072|
    +----------+----------+------+----------+-----+-------------------+
    only showing top 20 rows
    
    


```python
from pyspark.sql.functions import desc
from pyspark.sql.functions import col

sort_df = inner_join.filter("`P(A/B)` > 0.8")
sort_df.toPandas().to_csv("C:\\Users\\dpawa\\OneDrive\\Documents\\657\\assign_1_657_Pawar\\output\\probabilityandcoocc.csv")
```


```python

```


```python

```


```python

```
