# Setup ---------------------------------------------------------------
## Load Libs ---------------------------------------------------------------
library(tidyverse)
library(data.table)
library(slider, include.only = c('slide_chr', 'slide_index_chr'))
library(uwot, include.only = 'umap')
library(dbscan, include.only = c('hdbscan', 'dbscan'))
library(rnndescent)

dotenv::load_dot_env('.env')
db_helpers = local({source('./r_helpers/db-helpers.r', local = T); environment()})

model_prefix = 'olmoe'

## Helpers ---------------------------------------------------------------
big_format = scales::number_format(accuracy = 1e-2, scale_cut = scales::cut_short_scale())
get_entropy = function(counts) {
	if(sum(counts) == 0) return(0)
	p = counts / sum(counts)
	-sum(p * log2(p))
}

## Get Data ---------------------------------------------------------------
local({

	languages = c('en', 'cn', 'es')
	data_dir = 'analyses/path-analysis/data'

	# Get raw token-level df, replace lang x batch_ix x sequence_ix x token_ix identifier with sample_ix
	sample_df_raw =
		map(languages, \(l)
				fread(
					str_glue('{data_dir}/{model_prefix}-{l}-samples.csv'),
					colClasses = list(integer = c(1, 2, 3, 4, 7), character = 6, double = 5),
					strip.white = F,
					na.strings = NULL
					) %>%
					.[, lang := l]
				) %>%
		rbindlist() %>%
		.[, seq_id := .GRP, by = list(lang, batch_ix, sequence_ix)] %>%
		.[, is_seq_start := as.integer(token_ix == min(token_ix)), by = seq_id] %>%
		.[, sample_ix := .GRP, by = list(lang, batch_ix, sequence_ix, token_ix)] %>%
		.[, token := str_replace_all(token, '\n', '<linebreak>')] %>%
		arrange(., sample_ix) %>%
		select(
			.,
			sample_ix, seq_id, is_seq_start,
			lang, batch_ix, sequence_ix, token_ix,
			is_seq_start, token_id, token, output_id, output_prob
			)

	# Get raw sample x layer level dataframe
	routes_df =
		map(languages, \(l)
				fread(
					str_glue('{data_dir}/{model_prefix}-{l}-routes-top1.csv'),
					colClasses = list(integer = c(1, 2, 3, 4, 5, 7), double = 6),
					na.strings = ''
					) %>%
					.[, lang := l] %>%
					.[, layer_ix := layer_ix + 1] %>%
					.[, weight := NULL] %>%
					.[, topk_ix := NULL]
		) %>%
		rbindlist() %>%
		merge(
			.,
			select(sample_df_raw, sample_ix, lang, batch_ix, sequence_ix, token_ix),
			by = c('lang', 'batch_ix', 'sequence_ix', 'token_ix'),
			all = F
			) %>%
		select(., sample_ix, layer_ix, expert) %>%
		arrange(., sample_ix, layer_ix)

	n_layers = length(unique(routes_df$layer_ix))
	n_experts = length(unique(routes_df$expert))

	# Now create the string context around each sample, and additionally collapse the routes into a list-col
	sample_df =
		sample_df_raw %>%
		.[order(seq_id, sample_ix)] %>%
		.[,
			tokf := ifelse(is_seq_start == 1, paste0('<bos>', token), token),
			by = seq_id
			] %>%
		# Get the context around the token
		.[,
			pre := slide_index_chr(tokf, sample_ix, \(x) paste0(x, collapse = ''), .before = 8, .after = -1),
			by = seq_id
			] %>%
		.[,
			post := slide_index_chr(tokf, sample_ix, \(x) paste0(x, collapse = ''), .before = -1, .after = 4),
			by = seq_id
			] %>%
		.[, token_context := ifelse(
			is_seq_start == 1,
			paste0(pre, '<bos>[[', str_sub(tokf, 6), ']]', post),
			paste0(pre, '[[', tokf, ']]', post)
			)] %>%
		# Get the word that each token belongs to
		# New word boundary created whenever (new seq start) or (token has a space/linebreak/etc)
		# or (prev token ENDS with a a space/linebreak/etc)
		.[, word_ix := cumsum(fifelse(
			is_seq_start == 1 |
			str_detect(token, "\\s|[.,;:!?\\-\\(\\)\\[\\]\\{\\}/]|<linebreak>") |
			is.na(shift(token, 1)) |
			str_detect(shift(token, 1), "\\s$|[.,;:!?\\-\\(\\)\\[\\]\\{\\}/]$|<linebreak>$"),
			1, 0))
		] %>%
		.[
			,
			token_word := slider::slide_chr(tokf, \(x) paste0(x, collapse = ''), .before = 4, .after = 4),
			by = word_ix
		] %>%
		.[, c('pre', 'post', 'is_seq_start', 'tokf', 'word_ix') := NULL] %>%
		merge(
			.,
			routes_df %>% .[, list(route = list(expert)), by = list(sample_ix)],
			by = c('sample_ix'),
			all = F
			) %>%
		.[, c('batch_ix', 'sequence_ix', 'token_ix') := NULL]

	n_experts <<- n_experts
	n_layers <<- n_layers
	routes_df <<- as_tibble(routes_df)
	sample_df <<- as_tibble(sample_df)
})

# Data Checks  ---------------------------------------------

## Graph Expert Distribution ---------------------------------------------
local({

	routes_df %>%
		head(., 1e7) %>%
		count(expert, layer_ix) %>%
		filter(., layer_ix %in% c(0, 1, 5, 10, 15, n_layers)) %>%
		ggplot() +
		geom_col(aes(x = expert, y = n), position = 'identity', fill = 'skyblue') +
		facet_grid(rows = vars(layer_ix)) +
		labs(title = 'Frequency of experts as topk = 1 relative to other topks', x = 'Expert ID', y = 'Routing Counts')

	# By language
	bind_rows(
		head(filter(sample_df, lang == 'en'), 5e5),
		head(filter(sample_df, lang == 'cn'), 5e5)
		) %>%
		left_join(routes_df, by = 'sample_ix') %>%
		count(expert, layer_ix, lang) %>%
		filter(., layer_ix %in% c(0, 1, 5, 10, 15, n_layers)) %>%
		ggplot() +
		geom_col(aes(x = expert, y = n), position = 'identity', fill = 'skyblue') +
		facet_grid(rows = vars(layer_ix), cols = vars(lang)) +
		labs(title = 'Frequency of experts as topk = 1 relative to other topks', x = 'Expert ID', y = 'Routing Counts')

})

## Check Coverage ---------------------------------------------------
local({

	#' Print how many distinct routes are needed to cover the sample, conditional on different path lengths
	#'
	#' @param sample_df A sample-level df column `route`, a list-col of routed expert ids
	#' @param test_levels A nested list of vectors for all the levels you want to test
	#'
	#' @examples \dontrun{
	#'   # Print route coverage of layers 1:3, layers 1-3-5
	#'   test_levels = list(c(1:3), c(1, 3, 5))
	#'   print_path_counts(sample_df, test_levels)
	#' }
	#'
	#' @returns T
	print_path_counts = function(sample_df, test_levels) {
		print(str_glue('Sample tokens: {big_format(nrow(sample_df))}'))
		walk(test_levels, function(test_levs) {

			test_combo =
				sample_df %>%
				mutate(., route = map(route, \(x) x[test_levs])) %>%
				group_by(route) %>%
				summarize(n_tokens = n(), .groups = 'drop') %>%
				arrange(desc(n_tokens)) %>%
				mutate(., cumprop_toks = cumsum(n_tokens)/sum(n_tokens), cumprop_rtes = (1:n())/nrow(.))

			print(str_glue(
				'\nTest levels: {paste0(test_levs, collapse = ", ")}',
				'\n - Distinct routes: {big_format(nrow(test_combo))}',
				'\ | Routes to cover 95%: {big_format(nrow(filter(test_combo, cumprop_toks < .95)))}',
				'\ | Routes to cover 80%: {big_format(nrow(filter(test_combo, cumprop_toks < .8)))}'
			))
		})
	}

	# Test combinations of layers that return a reasonable number of paths
	print_path_counts(
		sample_df %>% filter(., lang == 'en'),
		list(
			c(1:3), c(2:4), c(1, 5, 10),
			c(1:4), c(1, 3, 5, 7), c(1:5),
			c(2:6), c(4:8),
			c(1:7), c(2:8), c(3:9), c(1, 3, 5, 7, 9, 11, 13),
			c(1:4, 8, 10, 12, 14)
			)
		)


	#' Print expert continuity by layer
	#'
	#' @param sample_df A sample-level df column `route`, a list-col of routed expert ids
	#'
	#' @returns A layer_ix level dataframe
	get_ec_by_layer = function(sample_df) {
		sample_df %>%
			select(., sample_ix, route) %>%
			unnest_longer(route, values_to = 'expert', indices_to = 'layer_ix') %>%
			group_by(., sample_ix) %>%
			arrange(., sample_ix, layer_ix, .by_group = T)  %>%
			mutate(., is_cont = ifelse(expert == lead(expert, 1), 1, 0)) %>%
			filter(., !is.na(is_cont)) %>%
			group_by(., layer_ix) %>%
			summarize(., ec = sum(is_cont)/n(), .groups = 'drop')
	}

	bind_rows(
		get_ec_by_layer(sample_df %>% filter(., lang == 'en')	%>% sample_n(., 10000)) %>% mutate(., lang = 'en'),
		get_ec_by_layer(sample_df %>% filter(., lang == 'cn')	%>% sample_n(., 10000)) %>% mutate(., lang = 'cn')
		) %>%
		ggplot() +
		geom_line(aes(x = layer_ix, y = ec, color = lang, group = lang))

	#' Get coverage stats by layer combination
	#'
	#' @param sample_df A sample-level df column `route`, a list-col of routed expert ids
	#' @param n_layers The number of layers routed to
	#' @param n_experts The number of experts routed to
	#' @param .verbose If T, prints progress
	#'
	#' @returns A dataframe at a (start_layer, end_layer) level with entropy metrics
	get_entropy_by_layer = function(sample_df, n_layers, n_experts, .verbose = F) {

		possible_layer_combs =
			expand_grid(start_layer = 1:n_layers, end_layer = 1:n_layers) %>%
			filter(end_layer > start_layer)

		route_level_df =
			sample_df %>%
			select(., sample_ix, route) %>%
			unnest_longer(route, values_to = 'expert', indices_to = 'layer_ix')

		layer_combos =
			possible_layer_combs %>%
			rowwise() %>%
			mutate(
				metrics = {

					if (.verbose) print(paste0(start_layer, '-', end_layer))
					path_length = end_layer - start_layer + 1

					# Convert to sample level
					subroutes_df =
						route_level_df %>%
						filter(layer_ix >= start_layer, layer_ix <= end_layer) %>%
						group_by(sample_ix) %>%
						arrange(layer_ix, .by_group = T) %>%
						summarize(route = list(expert), .groups = "drop")

					subroute_counts =
						subroutes_df %>%
						group_by(route) %>%
						summarize(n_tokens = n(), .groups = "drop") %>%
						arrange(desc(n_tokens)) %>%
						mutate(., cumprop_tokens = cumsum(n_tokens)/sum(n_tokens), cumprop_routes = (1:n())/nrow(.))

					obs_entropy = get_entropy(subroute_counts$n_tokens)
					max_entropy = path_length * log2(n_experts) # log2 = % of bits entropy
					norm_entropy = obs_entropy/max_entropy

					tibble(
						path_length = path_length,
						max_distinct_rtes = n_experts^path_length,
						n_distinct_rtes = nrow(subroute_counts),
						n_total_toks = sum(subroute_counts$n_tokens),
						t10_coverage = head(filter(subroute_counts, cumprop_routes > .1), 1)$cumprop_tokens,
						t50_coverage = head(filter(subroute_counts, cumprop_routes > .5), 1)$cumprop_tokens,
						obs_entropy = obs_entropy,
						max_entropy = max_entropy,
						norm_entropy = norm_entropy # Fraction of bits of capacity actually used
					)
				}
			) %>%
			ungroup() %>%
			unnest(cols = metrics)

		return(layer_combos)
	}

	entropy_df = get_entropy_by_layer(filter(sample_df, sample_ix <= 1e4), n_layers, n_experts, F)

	entropy_df %>%
		ggplot() +
		geom_tile(aes(x = start_layer, y = path_length, fill = t10_coverage)) +
		shadowtext::geom_shadowtext(
			aes(x = start_layer, y = path_length, label = scales::label_percent(accuracy = 1)(t10_coverage)),
			size = 3
		) +
		scale_x_continuous(breaks = 1:n_layers) +
		scale_fill_viridis_c(direction = 1) +
		theme_minimal() +
		labs(title = 'Token coverage from top 10% of paths', x = 'Starting Layer', y = 'Path Length') +
		theme(legend.position = 'none')

	entropy_df %>%
		ggplot() +
		geom_tile(aes(x = start_layer, y = path_length, fill = norm_entropy)) +
		shadowtext::geom_shadowtext(
			aes(x = start_layer, y = path_length, label = scales::label_percent(accuracy = 1)(norm_entropy)),
			size = 3
		) +
		scale_x_continuous(breaks = 1:n_layers) +
		scale_fill_viridis_c(direction = 1) +
		theme_minimal() +
		labs(title = 'Normalized entropy (fraction of bits of capacity used)', x = 'Starting Layer', y = 'Path Length') +
		theme(legend.position = 'none')


	# Test 2-layer path distributions
	expert_1_2_map =
		expand_grid(exp_1 = 0:(n_experts - 1), exp_2 = 0:(n_experts - 1)) %>%
		mutate(exp_id = 1:nrow(.))

	expert_1_2_df = list_rbind(map(list(c(1:2), c(10:11), c((n_layers - 1):n_layers)), function(l)
		sample_df %>%
			sample_n(5e5) %>%
			mutate(route = map(route, \(x) x[l[[1]]: l[[2]]])) %>%
			select(sample_ix, route) %>%
			unnest_longer(route, indices_to = 'index', values_to = 'exp') %>%
			pivot_wider(., names_from = 'index', values_from = 'exp', names_prefix = 'exp_') %>%
			inner_join(., expert_1_2_map, join_by(exp_1, exp_2)) %>%
			mutate(., start_end_layers = paste0(l[[1]], '-', l[[2]]))
	))

	expert_1_2_df %>%
		ggplot() +
		geom_density(aes(x = exp_id), bw = 0.1) +
		facet_grid(rows = vars(start_end_layers)) +
		theme_minimal() +
		labs(x = 'Expert cross-ID', y = 'Density', title = 'Expert two-layer routing probability')

	print_path_counts <<- print_path_counts
	get_entropy_by_layer <<- get_entropy_by_layer
})


# Get Clusters --------------------------------------------------------------

## Fixed Layer Indices --------------------------------------------------------------
local({

	#' Get basic path clusters by path size
	#'
	#' @param sample_level_df A sample-level dataframe with columns `route` (a list-col of routes), `token`,
	#'  `token_context`, `lang`, `output_id`
	#' @param grouping_layers A vector of integer indices for the layers to group columns by
	#' @param .entropy Whether to return the Shannon entropy of tokens in each group; useful for analyzing
	#'  token diversity
	#' @param .min_n_per_cluster The minimum size of each cluster. Lower returns smaller clusters, and is
	#'  more time costly.
	#'
	#' @returns A dataframe at a `path_str`/`subset_route` level, indicating a specific path.
	#'
	#' @export
	get_path_clusters = function(sample_level_df, grouping_layers, .entropy = F, .min_n_per_cluster = 20) {

		t0 = Sys.time()

		sample_level_df_segment =
			sample_level_df %>%
			mutate(., subset_route = map(route, \(x) x[grouping_layers]))

		paths =
			sample_level_df_segment %>%
			group_by(., subset_route) %>%
			summarize(., n_samples = n(), n_distinct_input_ids = n_distinct(token_id))

		valid_paths = filter(paths, n_samples >= .min_n_per_cluster)

		# Get sample-level stats seperately
		valid_path_sample_stats =
			sample_level_df_segment %>%
			filter(., subset_route %in% valid_paths$subset_route) %>%
			group_by(subset_route) %>%
			slice_sample(., n = 25) %>%
			summarize(
				.,
				
				token_samples = list(token),
				context_samples = list(token_context),
				word_samples = list(token_word),
				# primary_lang = count(cur_data(), lang) %>% filter(., n == max(n)) %>% pull(lang) %>% sample(., 1),
				lang_samples = list(lang),
				output_samples = list(output_id)
			)
		
		lang_freqs = 
			valid_path_sample_stats %>%
			select(subset_route, lang_samples) %>%
			unnest_longer(lang_samples, values_to = 'lang') %>% 
			count(subset_route, lang) %>%
			group_by(., subset_route) %>% 
			filter(., n == max(n)) %>%
			slice_sample(., n = 1) %>%
			transmute(subset_route, primary_lang = lang)

		valid_paths_final =
			valid_paths %>%
			inner_join(., valid_path_sample_stats, by = 'subset_route') %>%
			inner_join(., lang_freqs, by = 'subset_route')

		if (.entropy) {
			entropy_df =
				sample_level_df_segment %>%
				group_by(subset_route, token_id) %>%
				summarize(count = n(), .groups = 'drop') %>%
				group_by(subset_route) %>%
				summarize(entropy = get_entropy(count), .groups = 'drop')
		}

		res =
			valid_paths_final %>%
			transmute(
				.,
				subset_layers = list(grouping_layers),
				subset_route,
				path_str = map2_chr(subset_layers, subset_route, \(x, y) str_c(str_c('', x, ': ', y), collapse = ' -> ')),
				n_samples,
				n_distinct_input_ids,
				token_samples,
				context_samples,
				word_samples,
				primary_lang,
				lang_samples,
				output_samples
			) %>%
			{if (.entropy) inner_join(., entropy_df, join_by(subset_route)) else .}

		print(Sys.time() - t0)
		return(res)
	}

	#' Convert `path`-level df to CSV-friendly format
	#'
	#' @param paths_df A path-level df generated by `get_path_clusters`
	#'
	#' @export
	format_paths_csv = function(paths_df) {
		paths_df %>%
			mutate(
				across(c(context_samples), \(x) map_chr(x, \(y) paste0(y, collapse = '\n'))),
				across(
					c(token_samples, context_samples, word_samples, lang_samples, output_samples),
					\(x) map_chr(x, \(y) paste0(y, collapse = ','))
				)
			) %>%
			transmute(
				path_str,
				n_samples, n_distinct_input_ids,
				token_samples, context_samples, word_samples, primary_lang, lang_samples, output_samples
			)
	}

	#' Convert `path`-level df to SQL-friendly format
	#'
	#' @param paths_df A path-level df generated by `get_path_clusters`
	#'
	#' @export
	format_paths_sql = function(paths_df) {
		paths_df %>%
			mutate(
				across(
					c(subset_route, token_samples, context_samples, word_samples, lang_samples, output_samples),
					\(x) map_chr(x, \(y) jsonlite::toJSON(y, auto_unbox = T))
				)
			) %>%
			transmute(
				route = subset_route,
				n_samples,
				n_samples, n_distinct_input_ids,
				token_samples, context_samples, word_samples, primary_lang, lang_samples, output_samples
			)
	}

	
	paths_1_en = get_path_clusters(sample_df %>% filter(lang == 'en'), c(3:8), F, 10)
	
	paths_0_ml = get_path_clusters(sample_df %>% head(1000), c(3:8), F, 10)
	paths_1_ml = get_path_clusters(sample_df, c(2:8), F, 10)
	paths_2_ml = get_path_clusters(sample_df, c(5:9), F, 10)
	# paths_3_ml = get_path_clusters(sample_df, c(10:16), F, 10)
	# paths_4_ml = get_path_clusters(sample_df, c(16:20), F, 10)

	paths_1_ml %>% select(word_samples) %>% View()
	
	# Change model_cluster_method_id as well as ml
	paths_sql = paths_2_ml %>% format_paths_sql()

	sum(paths_sql$n_samples)
	nrow(sample_df)
	sum(paths_sql$n_samples)/nrow(sample_df)

	paths_sql %>%
		transmute(
			model_cluster_method_id = 6,
			route,
			token_samples,
			context_samples,
			word_samples,
			primary_lang,
			lang_samples,
			output_samples
		) %>%
		db_helpers$write_df_to_sql(
			pg,
			.,
			'paths',
			'ON CONFLICT (model_cluster_method_id, route) DO UPDATE SET 
			token_samples=EXCLUDED.token_samples,
			context_samples=EXCLUDED.context_samples,
			word_samples=EXCLUDED.word_samples,
			primary_lang=EXCLUDED.primary_lang,
			lang_samples=EXCLUDED.lang_samples,
			output_samples=EXCLUDED.output_samples',
			1000,
			T
		)

	paths_5_ml %>%
		format_paths_csv() %>%
		sample_n(10000) %>%
		write_csv(., 'qwen1.5moe-16-20.csv')
	
	test_z =
		sample_df %>%
		filter(str_detect(token_context, 'how dirty the nation')) 
	
	test_z %>% View()
	
	
})

## Individual Expert Analysis ----------------------------------------------
local({

	paths2 = get_path_clusters(sample_df_top1, c(2:4), .entropy = F)

	paths2 %>%
		filter(str_detect(path_str, '\\[l4\\] 1$')) %>%
		filter(str_detect(path_str, '\\[l2\\] 13 ->')) %>%
		View()

	paths3 = get_path_clusters(sample_df_top1, c(6:8), .entropy = F)

	paths3 %>%
		filter(str_detect(path_str, '\\[l4\\] 1$')) %>%
		filter(str_detect(path_str, '\\[l2\\] 13 ->')) %>%

	paths3 %>%
		filter(., str_detect(path_str, '\\[l8\\] 1')) %>%
		filter(., str_detect(path_str, '\\[l7\\] 1 ->'))



	sample_df_top1 %>%
		filter(., token_id == 27) %>%
		unnest_longer(route, indices_to = 'layer_id') %>%
		transmute(
			.,
			sample_ix,
			token_context,
			layer_id = layer_id - 1,s
			expert = route
			) %>%
		count(layer_id, expert) %>%
		View()
})



## Graph of all paths most strongly associated with token :
paths3 %>% filter(., str_detect(path_str, '\\[l8\\] 25$')) %>% View()



# Early Layer Path Predictability -----------------------------------------


# Get path predictability using first2 toks
uniq_paths =
	routes_df %>%
	filter(., layer_ix == 0 & topk_ix == 1) %>%
	head(., 1e7) %>%
	group_by(., seq_id) %>%
	mutate(., prev_token_id = lag(token_id, 1)) %>%
	ungroup() %>%
	filter(., is_start_of_seq != 1) %>%
	mutate(., tid_combo = paste0(prev_token_id, '-', token_id)) %>%
	group_by(., tid_combo, expert) %>%
	summarize(
		.,
		n_samples_for_tid_rte = n(),
		.groups = 'drop'
	)

most_common_per_group =
	uniq_paths %>%
	group_by(., tid_combo) %>%
	filter(., n_samples_for_tid_rte == max(n_samples_for_tid_rte)) %>%
	slice_head(., n = 1) %>%
	ungroup()




uniq_paths %>%
	View()

## Fixed Indices, Token Level ----------------------------------------------------
local({

	# View most common toks
	sample_df_top1 %>%
		count(token, token_id) %>%
		arrange(desc(n)) %>%
		view()

	test_token_df =
		sample_df_top1 %>%
		filter(., token == '-')

	print_path_counts(test_token_df, list(c(2:9), c(3:10), c(4, 6, 8, 10, 12, 14)))

	token_paths = get_path_clusters(test_token_df, c(4, 6, 8, 10, 12, 14), .entropy = F, .min_n_per_cluster = 10)

	token_paths %>% view()
})


## Variable Layers ---------------------------------------------------------
local({

	path_level_df =
		routes_df %>%
		filter(topk_ix == 1) %>%
		select(., -topk_ix, -is_start_of_seq, -token_id) %>%
		mutate(across(everything(), as.integer)) %>%
		group_by(sample_ix) %>%
		arrange(., layer_ix, .by_group = T) %>%
		mutate(
			expert_path = slide(expert, .f = \(x) x, .before = 0, .after = 5),
			layers = slide(layer_ix, .f = \(x) x, .before = 0, .after = 5)
		) %>%
		filter(lengths(expert_path) == 6) %>%
		transmute(., sample_ix, expert_path, layers)

	# Map samples
	token_sample_map =
		routes_df %>%
		filter(topk_ix == 1) %>%
		distinct(sample_ix, token_id, is_start_of_seq) %>%
		left_join(., vocab_df, by = 'token_id') %>%
		group_by(., sample_ix, token_id, token, is_start_of_seq) %>%
		summarize(., .groups = 'drop') %>%
		arrange(., sample_ix) %>%
		mutate(
			.,
			tok_fmt = ifelse(is_start_of_seq == 1, paste0('<eos>', token), token),
			prev_tok = slide_index_chr(tok_fmt, sample_ix, \(x) paste0(x, collapse = ''), .before = 6, .after = -1),
			next_tok = slide_index_chr(tok_fmt, sample_ix, \(x) paste0(x, collapse = ''), .before = -1, .after = 4),
			token_context = paste0(prev_tok, '[[', tok_fmt, ']] ', next_tok)
		)

	route_counts2 =
		path_level_df %>%
		left_join(., token_sample_map, by = 'sample_ix') %>%
		group_by(., expert_path, layers) %>%
		summarize(
			.,
			n_samples = n(),
			n_distinct_token_ids = n_distinct(token_id),
			token_samples = paste0(sample(token, min(length(token), 10)), collapse = ','),
			token_samples_with_context = paste0(sample(token_context, min(length(token_context), 10)), collapse = '\n'),
			.groups = 'drop'
		) %>%
		mutate(frac_distinct = n_distinct_token_ids/n_samples) %>%
		filter(., n_samples >= 10)

	route_counts2 %>%
		filter(., n_samples > 10 & frac_distinct >= .25) %>%
		View()

})


## Auto Clusters ---------------------------------------------
local({

	grouping_layers = 1:8
	cl_sample_size = 20000

	# View most common toks
	sample_df_top1 %>%
		count(token) %>%
		arrange(desc(n)) %>%
		view()

	test_token_df =
		sample_df_top1 %>%
		filter(., token == ' bank') %>%
		mutate(., route = map(route, \(x) x[grouping_layers]))

	test_token_df_wide =
		test_token_df %>%
		select(sample_ix, route) %>%
		unnest_wider(., route, names_sep = '_', names_repair = \(x) str_replace_all(x, 'route_', 'layer_'))

	# Clustering sample
	cluster_sample_df = sample_n(test_token_df_wide, min(nrow(test_token_df_wide), cl_sample_size))
	cluster_sample_mat = as.matrix(select(cluster_sample_df, starts_with('layer_')))

	umap_emb = umap(
		X = cluster_sample_mat,
		metric = 'hamming',
		nn_method = 'nndescent', # approximate nearest neighbors
		n_neighbors = 20, # Smaller more nuanced neighborhood size
		n_components = 2, # 2-dimensional
		# min_dist = 0.1, # tweak for more/less cluster separation
		verbose = T,
		ret_model = T,
		seed = 123
	)

	# kNNdistplot(umap_emb$embedding, k = 4)
	# dbscan_res = dbscan(umap_emb$embedding, eps = 1, minPts = 10)
	hdbscan_res = hdbscan(umap_emb$embedding, minPts = 30)

	cluster_sample_mat_clustered =
		cluster_sample_df %>%
		transmute(
			.,
			sample_ix,
			cluster_id = hdbscan_res$cluster,
			emb_1 = umap_emb$embedding[, 1],
			emb_2 = umap_emb$embedding[, 2]
			) %>%
		left_join(
			.,
			test_token_df,
			join_by('sample_ix')
		)

	# Token-level plot
	cluster_sample_mat_clustered %>%
		as_tibble() %>%
		mutate(., cluster_id = hdbscan_res$cluster) %>%
		ggplot() +
		geom_point(aes(x = emb_1, y = emb_2, color = as.factor(cluster_id))) +
		theme_minimal() +
		labs(
			labs = 'UMAP 2-dimensional path clusters for token "of"',
			color = 'Cluster ID',
			x = 'Dimension 1',
			y = 'Dimension 2'
			)

	cluster_sample_mat_clustered %>%
		mutate(
			.,
			layers = map(layers, \(x) x[grouping_layers]),
			path_str = map2_chr(layers, route, \(x, y) str_c(str_c('[l', x, '] ', y), collapse = ' -> '))
			) %>%
		group_by(cluster_id) %>%
		summarize(
			.,
			n_samples = n(),
			path_samples = paste0(sample(path_str, min(length(path_str), 50)), collapse = ','),
			token_samples = paste0(sample(token, min(length(token), 50)), collapse = ','),
			token_samples_with_context = paste0(sample(token_context, min(length(token_context), 50)), collapse = '\n'),
			.groups = 'drop'
		) %>%
		View()



		transmute(
			.,
			path_samples,
			n_samples,
			token_samples,
			token_samples_with_context
		) %>%
		write_csv(., str_glue('{model_prefix}-of.csv'))

})
