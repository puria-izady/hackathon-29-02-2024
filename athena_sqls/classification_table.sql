CREATE EXTERNAL TABLE IF NOT EXISTS classified_feedback(
  uid int,
  free_text varchar(255),
  classification varchar(255),
  confidence float
  )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '|'
ESCAPED BY '\\'
LINES TERMINATED BY '\n'
LOCATION 's3://YOUR-BUCKET/tenant-1/CLASSIFICATION/'