from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder.appName("CustomerIncomeReport").getOrCreate()

exchange_rates_df = spark.read.csv("exchange_rates.csv", header=True, inferSchema=True)

latest_rates_window = (
    Window
    .partitionBy("FROM_CURRENCY")
    .orderBy(F.desc("PARTITION_DATE"))
)

latest_rates_df = (
    exchange_rates_df.withColumn("rn", F.row_number().over(latest_rates_window))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

payments_df = spark.read.csv("payments.csv", header=True, inferSchema=True)

payments_df = (
    payments_df
    .withColumnsRenamed({column: column.strip() for column in payments_df.columns})
)

payments_with_exchange_rates_df = (
    payments_df
    .select(
        *[F.trim(column).alias(column) for column in payments_df.columns]
    )
    .filter(F.col("DATE") >= F.add_months(F.current_date(), -12))
    .withColumn("MONTH", F.date_format("DATE", "yyyy-MM"))
    .distinct()
    .alias("payments")
    .join(
        latest_rates_df.alias("exchange_rates"),
        on=F.col("payments.CURRENCY") == F.col("exchange_rates.FROM_CURRENCY"),
        how="left",
    )
    .withColumn("AMOUNT", F.col("payments.AMOUNT") * F.col("exchange_rates.RATE"))
)

# to store separatly until we receive rates for that payments
invalid_payments_df = (
    payments_with_exchange_rates_df
    .filter(F.col("exchange_rates.RATE").isNull())
)

report_df = (
    payments_with_exchange_rates_df
    .filter(F.col("exchange_rates.RATE").isNotNull())
    .groupby("payments.CUSTOMER_ID", "payments.COUNTRY", "payments.MONTH", "payments.PARTITION_DATE")
    .agg(
        F.sum("AMOUNT").cast("int").alias("TOTAL_PAYMENTS_EUR"),
        F.max("AMOUNT").cast("int").alias("MAX_PAYMENT_EUR")
    )
    .selectExpr(
        "CUSTOMER_ID",
        "COUNTRY",
        "MONTH",
        "TOTAL_PAYMENTS_EUR",
        "round(avg(TOTAL_PAYMENTS_EUR) over (partition by CUSTOMER_ID order by MONTH rows between 2 preceding and current row), 2) as ROLLING_3M_AVG_EUR",
        "MAX_PAYMENT_EUR",
        "dense_rank() over (partition by COUNTRY, MONTH order by TOTAL_PAYMENTS_EUR desc) as COUNTRY_RANK",
        "cast(PARTITION_DATE as date) as PARTITION_DATE",
    )
)

report_df.show()
