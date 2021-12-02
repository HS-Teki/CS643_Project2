import pyspark
import numpy
import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.ml.functions import array_to_vector
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
#`

'''
Columns from CSV:
'"""fixed acidity""""'
'""""volatile acidity""""'
'""""citric acid""""'
'""""residual sugar""""'
'""""chlorides""""'
'""""free sulfur dioxide""""'
'""""total sulfur dioxide""""'
'""""density""""'
'""""pH""""'
'""""sulphates""""'
'""""alcohol""""'
'""""quality"""""'
'''

def as_old(v):
    #print("Inside as_old")
    if isinstance(v, ml_linalg.SparseVector):
        #print("Sparse Converted")
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        #print("Dense Converted")
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))

'''
def parsePoint(line):
    return LabeledPoint(line[0], line[1:])
'''


print("Hello, world")
#spark = SparkContext(appName="WineQuality")
#spark = new SparkContext()
spark = SparkContext()
sc = SQLContext(spark)

#spark.setLogLevel('WARN')


path = "file:/home/ec2-user/TrainingDataset.csv"
path2 = "file:/home/ec2-user/ValidationDataset.csv"
#path3 = "file:/home/ec2-user/SVM/part-00000"
#path4 = "file:/home/ec2-user/TrainingDataset"
path4 = "file:/home/ec2-user/TrainingDataset"
#path5 = "/home/ec2-user/TrainingDataset"


df = sc.read.format('csv').load(path, inferSchema='true',header = True, sep=";")
dft = sc.read.format('csv').load(path2, inferSchema='true',header = True, sep=";")

df.cache()
df.show()
df.printSchema()

va = VectorAssembler(inputCols = ['"""""fixed acidity""""','""""volatile acidity""""','""""citric acid""""','""""residual sugar""""','""""chlorides""""','""""free sulfur dioxide""""','""""total sulfur dioxide""""','""""density""""','""""pH""""','""""sulphates""""','""""alcohol""""'], outputCol = 'features')
vdf = va.transform(df)
#vdf = vdf.select(['features','""""quality"""""'])
vdf = vdf.select(col('""""quality"""""').alias("label"), col("features"))
vdfp1 = vdf.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
print(vdfp1.take(10))

'''
va2 = VectorAssembler(inputCols = ['"""fixed acidity""""','""""volatile acidity""""','""""citric acid""""','""""residual sugar""""','""""chlorides""""','""""free sulfur dioxide""""','""""total sulfur dioxide""""','""""density""""','""""pH""""','""""sulphates""""','""""alcohol""""'], outputCol = 'features')
vdft = va2.transform(dft)
vdft = vdft.select(col('""""quality"""""').alias("label"), col("features"))
vdfp2 = vdft.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
print(vdfp2.take(10))
'''

cmd = 'rm -r TrainingDataset'
os.system(cmd)

#vdfSVM = MLUtils.saveAsLibSVMFile(vdfp1, path2)
#vdfLoad = MLUtils.loadLibSVMFile(spark,path3)

#parsedVDF = vdf.rdd.map(parsePoint)

#vdfp2 = vdfp1.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
#df_mat = RowMatrix(vdfp2)
#df_mat = IndexedRowMatrix(vdfp2)

#model = LogisticRegressionWithLBFGS.train(df_mat)

model = LogisticRegressionWithLBFGS.train(vdfp1, numClasses=10)
print("**********saving model**********")
model.save(spark,path4)
#model.save(vdfp1,path4)
print("**********loading model**********")
model = LogisticRegressionModel.load(spark,path4)

va2 = VectorAssembler(inputCols = ['"""fixed acidity""""','""""volatile acidity""""','""""citric acid""""','""""residual sugar""""','""""chlorides""""','""""free sulfur dioxide""""','""""total sulfur dioxide""""','""""density""""','""""pH""""','""""sulphates""""','""""alcohol""""'], outputCol = 'features')
vdft = va2.transform(dft)
vdft = vdft.select(col('""""quality"""""').alias("label"), col("features"))
vdfp2 = vdft.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
print(vdfp2.take(10))
predictionAndLabels = vdfp2.map(lambda lp: (float(model.predict(lp.features)), lp.label))

metrics = MulticlassMetrics(predictionAndLabels)

#precision = metrics.precision()
#recall = metrics.recall()
#f1Score = metrics.fMeasure(6.0)
#print("F1 Score = %s" % f1Score)

labels = vdfp2.map(lambda lp: lp.label).distinct().collect()
for label in sorted(labels):
    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

#labels = vdfp1.map(lambda lp: lp.label).distinct.collect()
#print("F1 Measure = %s" % metrics.fMeasure(vdf.label))
