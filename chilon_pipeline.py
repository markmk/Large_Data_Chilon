from pyspark.sql.functions import when, col, floor, expr
import datetime
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from sklearn.model_selection import GridSearchCV
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


df = spark.table("hive_metastore.default.delays__chilon_augmented")

# Define label (>= 15min or cancelled => 1)
df = df.withColumn(
    "label", when((col("CANCELLED") == 1) | (col("DEP_DELAY") >= 15), 1).otherwise(0)
)

delayed_fraction = df.filter(col("label") == 1).count() / df.count()

# Add the weight column to the entire df
df = df.withColumn(
    "classWeightCol",
    when(col("label") == 1, 1.0 / delayed_fraction).otherwise(1.0 / (1.0 - delayed_fraction))
)

df = df.na.fill({
    "WEATHER_DELAY_RATE_DEP_PREV_HOUR": 0.0,
    "WEATHER_DELAY_RATE_ARR_PREV_HOUR": 0.0,
    "z_score_dep_prev_hour": 0.0,
    "AVERAGE_SPEED": 0.0,
    "TOTAL_DELAYED_PER_CARRIER": 0.0
})

df = df.withColumn("SCHED_DEP_HOUR", floor(col("CRS_DEP_TIME") / 100))
df = df.na.fill({"SCHED_DEP_HOUR": 0.0})

# Time-based split for final validation (last 20% of days)
date_bounds = df.selectExpr("min(FL_DATE) as min_date", "max(FL_DATE) as max_date").collect()[0]
min_date, max_date = date_bounds["min_date"], date_bounds["max_date"]
total_days = (max_date - min_date).days
val_days = int(total_days * 0.2)
val_start = max_date - datetime.timedelta(days=val_days)

df_val = df.filter(col("FL_DATE") >= val_start)
df_remaining = df.filter(col("FL_DATE") < val_start)

train_df, test_df = df_remaining.randomSplit([0.8, 0.2], seed=42)
print("Training set:", train_df.count())
print("Test set:", test_df.count())
print("Final validation set:", df_val.count())

# Build pipeline stages for extra categorical features
day_indexer = StringIndexer(inputCol="DAY_OF_WEEK", outputCol="DAY_OF_WEEK_IDX", handleInvalid="keep")
day_encoder = OneHotEncoder(inputCol="DAY_OF_WEEK_IDX", outputCol="DAY_OF_WEEK_VEC")

season_indexer = StringIndexer(inputCol="SEASON", outputCol="SEASON_IDX", handleInvalid="keep")
season_encoder = OneHotEncoder(inputCol="SEASON_IDX", outputCol="SEASON_VEC")

assembler = VectorAssembler(
    inputCols=[
        "DAY_OF_WEEK_VEC",
        "SEASON_VEC",
        "WEATHER_DELAY_RATE_DEP_PREV_HOUR",
        "WEATHER_DELAY_RATE_ARR_PREV_HOUR",
        "z_score_dep_prev_hour",
        "AVERAGE_SPEED",
        "TOTAL_DELAYED_PER_CARRIER",
        "SCHED_DEP_HOUR"
    ],
    outputCol="features"
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeightCol",
    seed=42
)

pipeline_stages = [
    day_indexer, day_encoder,
    season_indexer, season_encoder,
    assembler,
    rf
]
pipeline = Pipeline(stages=pipeline_stages)

# Hyperparameter tuning with optimized GridSearchCV
paramGrid = (ParamGridBuilder()
  .addGrid(rf.maxDepth, [3, 5])
  .addGrid(rf.maxBins, [10, 20])
  .addGrid(rf.numTrees, [50, 100])
  .build()
)

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
    numFolds=5,
    seed=42
)

cv_model = crossval.fit(train_df)
bestModel = cv_model.bestModel

# Evaluate on test
predictions_test = bestModel.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc_test = evaluator.evaluate(predictions_test)
print("Test AUC after tuning:", auc_test)

# Evaluate on final validation
predictions_val = bestModel.transform(df_val)
auc_val = evaluator.evaluate(predictions_val)
print("Validation AUC after tuning:", auc_val)

predictions_val.groupBy("label", "prediction").count().show()
