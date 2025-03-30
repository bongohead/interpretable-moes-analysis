# Setup -------------------------------------------------------------------

## Load Libs ---------------------------------------------------------------
library(tidyverse)
library(DBI)
library(dotenv, include.only = 'load_dot_env')
load_dot_env()

## Connection --------------------------------------------------------------
pg = dbConnect(
	RPostgres::Postgres(),
	dbname = Sys.getenv('DB_NAME'),
	host = Sys.getenv('DB_HOST'),
	port = Sys.getenv('DB_PORT'),
	user = Sys.getenv('DB_USERNAME'),
	password = Sys.getenv('DB_PASSWORD')
)


# Create Tables -----------------------------------------------------------
local({
	
	dbExecute(pg, "DROP TABLE IF EXISTS models")
	dbExecute(pg, "DROP TABLE IF EXISTS cluster_methods")
	dbExecute(pg, "DROP TABLE IF EXISTS paths")
	
	
	dbExecute(pg, sql(
		'CREATE TABLE IF NOT EXISTS models (
			model_id SERIAL4 NOT NULL,
			model_name VARCHAR(255) NOT NULL,
			n_layers INT NOT NULL,		
			n_experts INT NOT NULL,
			description TEXT NULL,
			created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
			CONSTRAINT models_pk PRIMARY KEY (model_id)
		);'))
	
	dbExecute(pg, sql(
		'CREATE TABLE IF NOT EXISTS cluster_methods (
			cluster_method_id SERIAL4 NOT NULL,
			method_name VARCHAR(255) NOT NULL,
			layers VARCHAR(255) NULL,
			created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
			CONSTRAINT cluster_methods_pk PRIMARY KEY (cluster_method_id)
		);'))
	
	dbExecute(pg, sql(
		'CREATE TABLE IF NOT EXISTS paths (
			path_id SERIAL4 NOT NULL,
			model_id INT NOT NULL,
			cluster_method_id INT NOT NULL,
			route JSONB NOT NULL,
			token_samples JSONB NULL,
			context_samples JSONB NULL,
			primary_lang CHAR(2) NULL,
			lang_samples JSONB NULL,
			output_samples JSONB NULL,
			created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
			CONSTRAINT paths_pk PRIMARY KEY (path_id)
			CONSTRAINT paths_uk UNIQUE (model_id, cluster_method_id, route)
			CONSTRAINT forecast_values_v2_forecast_fk FOREIGN KEY (forecast) REFERENCES public.forecasts(id) ON DELETE CASCADE ON UPDATE CASCADE,
			CONSTRAINT forecast_values_v2_varname_fk FOREIGN KEY (varname) REFERENCES public.forecast_variables(varname) ON DELETE CASCADE ON UPDATE CASCADE
		);'))
	
	dbExecute(pg, sql('CREATE INDEX paths_model_id_ix ON paths USING btree (model_id);'))
	dbExecute(pg, sql('CREATE INDEX paths_cluster_method_id_ix ON paths USING btree (cluster_method_id);'))
	dbExecute(pg, sql('CREATE INDEX path_samples_context_trigram_ix ON paths USING GIN (samples_context gin_trgm_ops);'))
	
})

