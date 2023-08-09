CREATE TABLE IF NOT EXISTS public.t_measurement_item_id_relation (
id serial4 NOT NULL,
asset_id int4 NOT NULL,
unit int4 NOT NULL,
mill varchar NOT NULL,
roller int4 NOT NULL,
longitude int4 NOT NULL,
measurement_item_id int4 NOT NULL
);
