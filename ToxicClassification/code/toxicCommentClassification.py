from __future__ import print_function
import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="D:/Spark/spark-2.2.1-bin-hadoop2.7"
os.environ['HADOOP_HOME']="D:/Spark/spark-2.2.1-bin-hadoop2.7/hadoop"

# Append pyspark  to Python Path
sys.path.append("D:/Spark/spark-2.2.1-bin-hadoop2.7/python")
sys.path.append("D:/Spark/spark-2.2.1-bin-hadoop2.7/python/lib/py4j-0.10.4-src.zip")
try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)



import sys
from operator import add
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import SparkConf, SparkContext, HiveContext
import sys
import pandas as pd
import re
import string
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import udf
from pyspark.ml.classification import RandomForestClassifier

# spark = SparkSession \
#     .builder \
#     .appName("PythonWordCount") \
#     .getOrCreate()
# hc = HiveContext(spark)
conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '4G')
        .set('spark.driver.maxResultSize', '6G'))
sc = SparkContext(conf=conf)
hc = HiveContext(sc)



def to_spark_df(fin):
    df = pd.read_csv(fin)
    df.fillna("", inplace=True)
    df = hc.createDataFrame(df)
    return df


data = to_spark_df(sys.argv[1])
from pyspark.ml.classification import RandomForestClassifier

(train, test) = data.randomSplit([0.7, 0.3], seed=10)


import pandas as pd

df = test.toPandas()
df_train = train.toPandas()
df.to_csv('trainData.csv', sep=',', encoding='utf8')
df.to_csv('testData.csv', sep=',', encoding='utf8')

cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]

patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    return str(clean, 'utf-8')


F1 = udf(prepare_for_char_n_gram)
train = train.withColumn("comment_text", F1(train["comment_text"]))
test = test.withColumn("comment_text", F1(test["comment_text"]))

out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]

# Basic sentence tokenizer , We split each sentence into words using Tokenizer.
tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
wordsData = tokenizer.transform(train)

remover = StopWordsRemover(inputCol="words", outputCol="important_words")
wordsData = remover.transform(wordsData)

# Count the words in a document , For each sentence (bag of words), we use HashingTF to hash the sentence into a feature vector.
# HashingTF takes an RDD of list as the input.
hashingTF = HashingTF(inputCol="important_words", outputCol="rawFeatures")
tf = hashingTF.transform(wordsData)

# Build the idf model and transform the original token frequencies into their tf-idf counterparts  , We use IDF to rescale the feature vectors; this generally improves performance when using text as features.
# Our feature vectors could then be passed to a learning algorithm
# While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
# First to compute the IDF vector and second to scale the term frequencies by IDF.
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

#  logistic regression can be used to predict a binary outcome by using binomial logistic regression.
#  Build a logistic regression model for the binary toxic column.
#  Use the features column (the tfidf values) as the input vectors,  X, and the toxic column as output vector, y.
REG = 0.1
# lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)
# lrModel = lr.fit(tfidf)
# res_train = lrModel.transform(tfidf)

extract_prob = F.udf(lambda x: x)

test_tokens = tokenizer.transform(test)
wordsData = remover.transform(test_tokens)

test_tf = hashingTF.transform(wordsData)
test_tfidf = idfModel.transform(test_tf)

test_res = test.select('id')

test_probs = []


# Logistic Regression working code
if sys.argv[2] == 'LR':
    print('Prediction with Logistic egression')
    for col in out_cols:
        print(col)
        lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
        print("...fitting")
        lrModel = lr.fit(tfidf)
        print("...predicting")
        res = lrModel.transform(test_tfidf)
        print("...appending result")
        test_res = test_res.join(res.select('id','prediction'), on="id")
        print("...extracting probability")
        test_res = test_res.withColumn(col, extract_prob('prediction')).drop("prediction")
        # test_res.show()


if sys.argv[2] == 'RF':
    print('Prediction with Random Forest')
    for col in out_cols:
        print(col)
        rf = RandomForestClassifier(labelCol=col, featuresCol="features", numTrees=10)
        print("...fitting")
        rfModel = rf.fit(tfidf)
        print("...predicting")
        res = rfModel.transform(test_tfidf)
        print("...appending result")
        test_res = test_res.join(res.select('id','prediction'), on="id")
        print("...extracting probability")
        test_res = test_res.withColumn(col, extract_prob('prediction')).drop("prediction")
        # test_res.show()

test_res_pan = test_res.toPandas()
test_res_pan.to_csv("toxic_classification1.csv", index=False)
# test_res_pan_rf = test_res.toPandas()
# test_res_pan_rf.to_csv("toxic_classification1_rf.csv", index=False)