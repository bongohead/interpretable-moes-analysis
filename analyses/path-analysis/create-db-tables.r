# Setup -------------------------------------------------------------------

## Load Libs ---------------------------------------------------------------
library(tidyverse)
library(DBI)

dotenv::load_dot_env('.env')
db_helpers = local({source('./r_helpers/db-helpers.r', local = T); environment()})

## Connection --------------------------------------------------------------
pg = db_helpers$connect_pg()

# Create Tables -----------------------------------------------------------
local({
	
	# dbExecute(pg, "DROP TABLE IF EXISTS models")
	# dbExecute(pg, "DROP TABLE IF EXISTS cluster_methods")
	# dbExecute(pg, "DROP TABLE IF EXISTS paths")
	
	dbExecute(pg, sql(
		'CREATE TABLE IF NOT EXISTS models (
			id SERIAL4 NOT NULL,
			model_shortname VARCHAR(255) NOT NULL,
			model_fullname VARCHAR(255) NOT NULL,
			n_layers INT NOT NULL,		
			n_experts INT NOT NULL,
			description TEXT NULL,
			created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
			CONSTRAINT models_pk PRIMARY KEY (id)
		);'))
	
	
	dbExecute(pg, sql(
		'CREATE TABLE IF NOT EXISTS model_cluster_methods (
			id SERIAL4 NOT NULL,
			model_id INT NOT NULL,
			method_name VARCHAR(255) NOT NULL,
			dataset VARCHAR(255) NOT NULL,
			layers JSONB NULL,
			coverage_rate NUMERIC(5,4) NULL,
			coverage_count INT NULL,
			dataset_count INT NULL,
			created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
			CONSTRAINT model_cluster_methods_pk PRIMARY KEY (id),
			CONSTRAINT model_cluster_methods_uk UNIQUE (model_id, method_name, dataset),
			CONSTRAINT model_cluster_methods_model_id_fk FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE ON UPDATE CASCADE
		);'))
	
	
	dbExecute(pg, sql(
		'CREATE TABLE IF NOT EXISTS paths (
			path_id SERIAL4 NOT NULL,
			model_cluster_method_id INT NOT NULL,
			route JSONB NOT NULL,
			token_samples JSONB NULL,
			context_samples JSONB NULL,
			word_samples JSONB NULL,
			primary_lang CHAR(2) NULL,
			created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
			CONSTRAINT paths_pk PRIMARY KEY (path_id),
			CONSTRAINT paths_uk UNIQUE (model_cluster_method_id, route),
			CONSTRAINT paths_model_cluster_method_id_fk FOREIGN KEY (model_cluster_method_id) REFERENCES model_cluster_methods(id) ON DELETE CASCADE ON UPDATE CASCADE
		);'))
	
	dbExecute(pg, sql('CREATE INDEX paths_model_cluster_method_id_ix ON paths USING btree (model_cluster_method_id);'))
	dbExecute(pg, sql('CREATE INDEX paths_token_samples_trigram_ix ON paths USING GIN ((token_samples::TEXT) gin_trgm_ops);'))
	dbExecute(pg, sql('CREATE INDEX paths_context_samples_trigram_ix ON paths USING GIN ((context_samples::TEXT) gin_trgm_ops);'))
	dbExecute(pg, sql('CREATE INDEX paths_word_samples_trigram_ix ON paths USING GIN ((word_samples::TEXT) gin_trgm_ops);'))
})



paths_1_sql %>%
	transmute(
		model_id = 1,
		cluster_method_id = 1,
		route,
		token_samples,
		context_samples,
		word_samples,
		primary_lang,
		lang_samples,
		output_samples
	) %>%
	write_df_to_sql(
		pg,
		.,
		'paths',
		'ON CONFLICT (model_id, cluster_method_id, route) DO UPDATE SET 
			token_samples=EXCLUDED.token_samples,
			context_samples=EXCLUDED.context_samples,
			word_samples=EXCLUDED.word_samples,
			primary_lang=EXCLUDED.primary_lang,
			lang_samples=EXCLUDED.lang_samples,
			output_samples=EXCLUDED.output_samples',
		1000,
		T
		)

