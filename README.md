# Bigdata analysis using PySpark: XGBoost_model

![Banner](docs/assets/images/banner_bigdata.jpg)

### Python code:

### 1. Import libraries
```
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from sparkxgb import XGBoostRegressor
```
### 2. Initialize Spark session for distributed computation
```
spark = SparkSession.builder.appName("XGBoostRegressionExample").getOrCreate()
```
### 3. Load the big dataset into a DataFrame. Use a distributed format like CSV or Parquet
```
data = spark.read.csv(‘datafile.scv’, header=True, inferSchema=True)
```
### 4. Features & label. `VectorAssembler` combines all feature columns into a single vector column required by MLlib
```
feature_cols = [col for col in data.columns if col != "label"]    # feature_cols = [‘’v1”, “v2”, “v3”, “v4”, “v5”]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
```
### 5. Transform data before splitting
```
assembled_data = assembler.transform(data)
```
### 6. Split data into train and test sets
```
train, test = assembled_data.randomSplit([0.8, 0.2], seed=42)
```
### 7. Create XGBoost regressor. Requires spark-xgb package
```
xgb = XGBoostRegressor(
    featuresCol="features",
    labelCol="label",
    maxDepth=6,
    n_estimators=100,
    objective="reg:squarederror"
)
```
### 8. Build pipeline. Combines preprocessing and modeling into a single pipeline for reproducibility
```
pipeline = Pipeline(stages=[assembler, xgb])
```
### 9. Train the model. Fits the pipeline to the training data
```
model = pipeline.fit(train)
```
### 10. Predict on test set
```
predictions = model.transform(test)
```
### 11. Evaluate model
```
evaluator_rmse = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"                           )
rmse = evaluator_rmse.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
```
### 12. Stop PySpark
```
spark.stop()
```
