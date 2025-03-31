#' Connect to a Postgres database
#'
#' @importFrom RPostgres Postgres
#' @importFrom DBI dbConnect
#'
#' @export
connect_pg = function() {

	db = dbConnect(
		RPostgres::Postgres(),
		dbname = Sys.getenv('DB_NAME'),
		host = Sys.getenv('DB_HOST'),
		port = Sys.getenv('DB_PORT'),
		user = Sys.getenv('DB_USERNAME'),
		password = Sys.getenv('DB_PASSWORD')
	)

	return(db)
}

#' Execute a select query
#'
#' @param db The DBI connector object
#' @param query The query
#'
#' @importFrom DBI dbGetQuery
#' @importFrom purrr is_scalar_character
#'
#' @export
get_query = function(db, query) {

	if (!inherits(db, 'PqConnection')) stop('Object "db" must be of class PgConnection')
	if (!is_scalar_character(query)) stop('Parameter "query" must be a character!')

	return(as_tibble(dbGetQuery(db, query)))
}

#' Disconnect from DB
#'
#' @param db The DBI connector object
#'
#' @importFrom DBI dbDisconnect
#'
#' @export
disconnect_db = function(db) {

	if (!inherits(db, 'PqConnection')) stop('Object "db" must be of class PgConnection')

	dbDisconnect(db)
}

#' Check largest table sizes in Postgres database.
#'
#' @param db A PostgreSQL DBI connection.
#'
#' @return A data frame of table sizes.
#'
#' @import dplyr
#'
#' @export
get_pg_table_sizes = function(db) {

	if (!inherits(db, 'PqConnection')) stop('Object "db" must be of class PgConnection')

	res = get_query(db, sql(
		"SELECT
			schema_name,
			relname,
			pg_size_pretty(table_size) AS size,
			table_size
		FROM (
			SELECT
				pg_catalog.pg_namespace.nspname AS schema_name,
				relname,
				pg_relation_size(pg_catalog.pg_class.oid) AS table_size
			FROM pg_catalog.pg_class
			JOIN pg_catalog.pg_namespace ON relnamespace = pg_catalog.pg_namespace.oid
		) t
		WHERE schema_name NOT LIKE 'pg_%'
		ORDER BY table_size DESC;"
	))
}

#' Helper function to check row count of a table in Postgres
#'
#' @description
#' Returns the number of rows in a table
#'
#' @param db The database connection object.
#' @param tablename The name of the table.
#'
#' @importFrom DBI dbGetQuery
#' @importFrom purrr is_scalar_character
#'
#' @export
get_rowcount = function(db, tablename) {

	if (!inherits(db, 'PqConnection')) stop('Object "db" must be of class PgConnection')
	if (!is_scalar_character(tablename)) stop('Parameter "tablename" must be a character!')

	count = as.numeric(dbGetQuery(db, paste0('SELECT COUNT(*) AS count FROM ', tablename, ''))$count)

	return(count)
}

#' Create a SQL INSERT query string.
#'
#' @param df A data frame to insert into SQL.
#' @param tblname The name of the SQL table.
#' @param .append Any string to append at the end of the query; useful for ON DUPLICATE statements.
#'
#' @return A character string representing a SQL query.
#'
#' @import dplyr
#' @importFrom tidyr unite
#' @importFrom lubridate is.POSIXct
#' @importFrom DBI dbQuoteString ANSI
#' @importFrom purrr is_scalar_character
#'
#' @export
create_insert_query = function(df, tblname, .append = '') {

	if (!is.data.frame(df)) stop('Parameter "df" must be a data.frame.')
	if (!is_scalar_character(.append)) stop('Parameter ".append" must be a character.')

	paste0(
		'INSERT INTO ', tblname, ' (', paste0(colnames(df), collapse = ','), ')\n',
		'VALUES\n',
		df %>%
			mutate(across(where(is.Date), as.character)) %>%
			mutate(across(where(is.POSIXct), function(x) format(x, '%Y-%m-%d %H:%M:%S %Z'))) %>%
			mutate(across(where(is.character), function(x) dbQuoteString(ANSI(), x))) %>%
			mutate(across(where(is.numeric), function(x) dbQuoteString(ANSI(), as.character(x)))) %>%
			unite(., 'x', sep = ',') %>%
			mutate(., x = paste0('(', x, ')')) %>%
			.$x %>%
			paste0(., collapse = ', '), '\n',
		.append, ';'
	) %>%
		return(.)
}

#' Inserts a dataframe into SQL
#'
#' @description
#' Writes a dataframe into SQL using an INSERT query.
#'
#' @param df A data frame to insert into SQL.
#' @param tblname The name of the SQL table.
#' @param .append Any additional characters to append to the end of the query.
#' @param .chunk_size The number of rows to send at a time.
#' @param .verbose Whether to log progress of the SQL write.
#'
#' @return A count of the number of rows added.
#'
#' @examples \dontrun{
#' pg = connect_pg()
#' # Create a table with a unique key
#' DBI::dbExecute(
#'   pg,
#' 	 "CREATE TABLE test (varname TEXT,  date TEXT,  value DECIMAL, UNIQUE(varname, date))"
#'   )
#' to_write =
#' 	 get('economics') %>%
#' 	 pivot_longer(., -date, names_to = 'varname', values_to = 'value')
#'
#' # Test writing everything except pce
#' rows_written = write_df_to_sql(pg, filter(to_write, varname != 'pce'), 'test', .append = 'ON CONFLICT (varname, date) DO NOTHING')
#' print(rows_written)
#'
#' # Test now writing the entire table with pce - only the pce rows should insert
#' rows_written = write_df_to_sql(pg, to_write, 'test', .append = 'ON CONFLICT (varname, date) DO NOTHING')
#' print(rows_written)
#'
#' DBI::dbExecute(pg, 'DROP TABLE test')
#' }
#'
#' @import dplyr
#' @importFrom purrr map_dbl is_scalar_logical is_scalar_double is_scalar_integer is_scalar_character
#' @importFrom DBI dbExecute
#'
#' @export
write_df_to_sql = function(db, df, tblname, .append = '', .chunk_size = 1000, .verbose = F) {

	if (!inherits(db, 'PqConnection')) stop('Object "db" must be of class PgConnection')
	if (!is.data.frame(df)) stop('Parameter "df" must be a data.frame.')
	if (!is_scalar_character(tblname)) stop('Parameter "tblname" must be a character.')
	if (!is_scalar_character(.append)) stop('Parameter ".append" must be a character.')
	if (!is_scalar_double(.chunk_size) && !is_scalar_integer(.chunk_size)) stop('Parameter ".chunk_size" must be an integer.')
	if (!is_scalar_logical(.verbose)) stop('Parameter ".verbose" must be a logical.')

	rows_modified_by_chunk =
		df %>%
		mutate(., split = ceiling((1:nrow(df))/.chunk_size)) %>%
		group_split(., split, .keep = FALSE) %>%
		map_dbl(., .progress = .verbose, function(x) {
			dbExecute(db, create_insert_query(x, tblname = tblname, .append = .append))
		})

	if (any(is.null(rows_modified_by_chunk))) {
		stop('SQL Error!')
	}

	return(sum(rows_modified_by_chunk))
}
