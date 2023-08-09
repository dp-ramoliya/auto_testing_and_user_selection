CREATE TABLE IF NOT EXISTS t_iot_data (
plant_id VARCHAR(10) NOT NULL,
unit_id VARCHAR(10) NOT NULL,
msumt_datetime timestamp NOT NULL,
t_pid_no_text VARCHAR(255) NOT NULL,
quality_number integer NOT NULL,
msumt_value double precision NOT NULL,
create_at timestamp NOT NULL,
update_at timestamp NOT NULL,
process_at timestamp NOT NULL,
process_id VARCHAR(255) NOT NULL
);
