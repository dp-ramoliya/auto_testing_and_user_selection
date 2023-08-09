-- Fetch total current sensor counts
SELECT COUNT(*)
FROM t_iot_data
JOIN t_mill_model_pid_relation ON t_iot_data.t_pid_no_text = t_mill_model_pid_relation.t_pid_no_text
WHERE t_mill_model_pid_relation.asset_id = {asset_id}
AND t_mill_model_pid_relation.sensor_name = 'current'
AND t_iot_data.msumt_datetime >= '{highest_date}'
AND t_iot_data.msumt_datetime <= '{current_date}'
AND t_iot_data.msumt_value >= 1
AND t_iot_data.msumt_value <= 200