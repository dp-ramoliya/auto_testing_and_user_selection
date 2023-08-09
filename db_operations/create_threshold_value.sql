CREATE TABLE IF NOT EXISTS public.t_threshold_value (
id serial4 NOT NULL,
unit int4 NOT NULL,
threshold int4 NOT NULL,
UNIQUE (unit)
);
