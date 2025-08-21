import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, Window


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("CustomerIncomeReportTest") \
        .getOrCreate()
    yield spark
    spark.stop()


def test_unique_payments_id(spark):
    data = [
        (1, "C1", "PL", 1000, "PLN", "2025-06-10", "2025-08-17"),
        (2, "C1", "PL", 2000, "PLN", "2025-07-05", "2025-08-17"),
        (2, "C1", "PL", 2000, "PLN", "2025-07-05", "2025-08-17"),
        (3, "C1", "PL", 500, "PLN", "2025-08-01", "2025-08-17"),
        (4, "C2", "PL", 3000, "PLN", "2025-07-15", "2025-08-17"),
        (5, "C3", "DE", 1000, "EUR", "2025-08-05", "2025-08-17"),
        (6, "C1", "PL", 5000, "PLN", "2025-07-14", "2025-08-17")
    ]
    columns = ["PAYMENT_ID", "CUSTOMER_ID", "COUNTRY", "AMOUNT", "CURRENCY", "DATE", "PARTITION_DATE"]

    df = spark.createDataFrame(data, columns)

    unique_df = df.dropDuplicates(["PAYMENT_ID"])

    payment_ids = [row["PAYMENT_ID"] for row in unique_df.select("PAYMENT_ID").collect()]
    assert len(payment_ids) == len(set(payment_ids)), "Duplicates were found based on PAYMENT_ID"


def test_last_twelve_months_payments(spark):
    data = [
        (1, "C1", "PL", 1000, "PLN", "2022-06-10", "2025-08-17"),
        (2, "C1", "PL", 2000, "PLN", "2023-07-05", "2025-08-17"),
        (3, "C1", "PL", 500, "PLN", "2025-08-01", "2025-08-17"),
        (4, "C2", "PL", 3000, "PLN", "2021-07-15", "2025-08-17"),
        (5, "C3", "DE", 1000, "EUR", "2025-08-05", "2025-08-17"),
        (6, "C1", "PL", 5000, "PLN", "2025-07-14", "2025-08-17")
    ]
    columns = ["PAYMENT_ID", "CUSTOMER_ID", "COUNTRY", "AMOUNT", "CURRENCY", "DATE", "PARTITION_DATE"]

    df = spark.createDataFrame(data, columns)

    df = df.withColumn("DATE", F.to_date("DATE"))

    last_twelve_months_df = (df.filter(F.col("DATE") >= F.add_months(F.current_date(), -12)))

    result_ids = [raw["PAYMENT_ID"] for raw in last_twelve_months_df.select("PAYMENT_ID").collect()]
    assert set(result_ids) == {3, 5, 6}, f"Unexpected payments in result: {result_ids}"


def test_total_payments_eur(spark):
    payments_data = [
        (1, "C1", "PL", 1000, "PLN", "2025-06-10", "2025-08-17"),
        (2, "C1", "PL", 2000, "PLN", "2025-07-05", "2025-08-17"),
        (3, "C1", "PL", 500, "PLN", "2025-08-01", "2025-08-17"),
        (4, "C2", "PL", 3000, "PLN", "2025-07-15", "2025-08-17"),
        (5, "C3", "DE", 1000, "EUR", "2025-08-05", "2025-08-17"),
        (6, "C1", "PL", 5000, "PLN", "2025-07-14", "2025-08-17")
    ]
    payments_data_columns = ["PAYMENT_ID", "CUSTOMER_ID", "COUNTRY", "AMOUNT", "CURRENCY", "DATE", "PARTITION_DATE"]

    payments_df = spark.createDataFrame(payments_data, payments_data_columns)

    exchange_rates_data = [
        ("PLN", "EUR", 0.22, "2025-08-17"),
        ("EUR", "EUR", 1.0, "2025-08-17")
    ]
    exchange_rates_columns = ["FROM_CURRENCY", "TO_CURRENCY", "RATE", "PARTITION_DATE"]

    exchange_rates_df = spark.createDataFrame(exchange_rates_data, exchange_rates_columns)

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

    payments_with_exchange_rates_df = (
        payments_df
        .distinct()
        .alias("payments")
        .join(
            latest_rates_df.alias("exchange_rates"),
            on=F.col("payments.CURRENCY") == F.col("exchange_rates.FROM_CURRENCY"),
            how="left",
    )
    .withColumn("AMOUNT", F.col("payments.AMOUNT") * F.col("exchange_rates.RATE"))
    )

    report_df = (
        payments_with_exchange_rates_df.filter(F.col("RATE").isNotNull())
        .groupby("payments.CUSTOMER_ID", "payments.COUNTRY", "payments.PARTITION_DATE")
        .agg(
            F.sum("AMOUNT").cast("int").alias("TOTAL_PAYMENTS_EUR")
        )
        .selectExpr(
            "CUSTOMER_ID",
            "TOTAL_PAYMENTS_EUR"
        )
    )

    result = report_df.select("CUSTOMER_ID", "TOTAL_PAYMENTS_EUR").collect()
    result_dict = {row["CUSTOMER_ID"]: row["TOTAL_PAYMENTS_EUR"] for row in result}

    expected = {
        "C1": 1870,
        "C2": 660,
        "C3": 1000
    }

    assert result_dict == expected, f"Unexpected aggregation result: {result_dict}"
     

def test_rolling_3m_avg_eur(spark):
    payments_data = [
        (1, "C1", "PL", 1000, "PLN", "2025-06-10", "2025-08-17"),
        (2, "C1", "PL", 2000, "PLN", "2025-07-05", "2025-08-17"),
        (3, "C1", "PL", 500, "PLN", "2025-08-01", "2025-08-17"),
        (4, "C2", "PL", 3000, "PLN", "2025-07-15", "2025-08-17"),
        (5, "C3", "DE", 1000, "EUR", "2025-08-05", "2025-08-17"),
        (6, "C1", "PL", 5000, "PLN", "2025-07-14", "2025-08-17")
    ]
    payments_data_columns = ["PAYMENT_ID", "CUSTOMER_ID", "COUNTRY", "AMOUNT", "CURRENCY", "DATE", "PARTITION_DATE"]

    payments_df = spark.createDataFrame(payments_data, payments_data_columns)

    exchange_rates_data = [
        ("PLN", "EUR", 0.22, "2025-08-17"),
        ("EUR", "EUR", 1.0, "2025-08-17")
    ]
    exchange_rates_columns = ["FROM_CURRENCY", "TO_CURRENCY", "RATE", "PARTITION_DATE"]

    exchange_rates_df = spark.createDataFrame(exchange_rates_data, exchange_rates_columns)

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

    payments_with_exchange_rates_df = (
        payments_df
        .distinct()
        .withColumn("MONTH", F.date_format("DATE", "yyyy-MM"))
        .alias("payments")
        .join(
            latest_rates_df.alias("exchange_rates"),
            on=F.col("payments.CURRENCY") == F.col("exchange_rates.FROM_CURRENCY"),
            how="left",
        )
    .withColumn("AMOUNT", F.col("payments.AMOUNT") * F.col("exchange_rates.RATE"))
    )

    report_df = (
        payments_with_exchange_rates_df.filter(F.col("RATE").isNotNull())
        .groupby("payments.CUSTOMER_ID", "payments.COUNTRY", "payments.MONTH", "payments.PARTITION_DATE")
        .agg(
            F.sum("AMOUNT").cast("int").alias("TOTAL_PAYMENTS_EUR")
        )
        .selectExpr(
            "CUSTOMER_ID",
            "MONTH",
            "round(avg(TOTAL_PAYMENTS_EUR) over (partition by CUSTOMER_ID order by MONTH rows between 2 preceding and current row), 2) as ROLLING_3M_AVG_EUR"
        )
    )

    result = report_df.select("CUSTOMER_ID", "MONTH", "ROLLING_3M_AVG_EUR").collect()

    result_data = [row.asDict() for row in result]

    expected_data = [
        {"CUSTOMER_ID": "C1", "MONTH": "2025-06", "ROLLING_3M_AVG_EUR": 220.0},
        {"CUSTOMER_ID": "C1", "MONTH": "2025-07", "ROLLING_3M_AVG_EUR": 880.0},
        {"CUSTOMER_ID": "C1", "MONTH": "2025-08", "ROLLING_3M_AVG_EUR": 623.33},
        {"CUSTOMER_ID": "C2", "MONTH": "2025-07", "ROLLING_3M_AVG_EUR": 660.0},
        {"CUSTOMER_ID": "C3", "MONTH": "2025-08", "ROLLING_3M_AVG_EUR": 1000.0}
    ]

    assert result_data == expected_data, f"Unexpected aggregation result: {result_data}"
    