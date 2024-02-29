CREATE EXTERNAL TABLE IF NOT EXISTS cost_table(
  team_id varchar(255),
  model_id varchar(255),
  input_tokens int,
  output_tokens int,
  input_cost float,
  output_cost float,
  invocations int,
  `date` date
  )
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
ESCAPED BY '\\'
LINES TERMINATED BY '\n'
LOCATION 's3://YOUR-BUCKET/tenant-1/TEST/'