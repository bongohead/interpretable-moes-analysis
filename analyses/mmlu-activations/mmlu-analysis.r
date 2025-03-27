# Setup ---------------------------------------------------------------
## Load Libs ---------------------------------------------------------------
library(tidyverse)
library(slider)
library(uwot, include.only = 'umap')
library(dbscan, include.only = c('hdbscan', 'dbscan'))
library(rnndescent)

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
	
	vocab_df =
		read_csv(str_glue('analyses/mmlu-activations/data/{model_prefix}-vocab.csv'), trim_ws = F) %>%
		transmute(., token_id, token) %>%
		mutate(token = str_replace_all(token, c(
			'âĢĻ|âĢľ' = '\'',
			'âĢĵ' = '-',
			'Ċ' = ' ',
			'âĢĿ|âĢĺ' = '"')
		))
	
	question_df_raw =
		read_csv(str_glue('analyses/mmlu-activations/data/{model_prefix}-mmlu-questions.csv')) %>%
		rename(., sample_ix = q_ix)

	routes_df =
		read_csv(str_glue('analyses/mmlu-activations/data/{model_prefix}-mmlu-topk.csv')) %>%
		rename(., sample_ix = q_ix)

	n_layers = length(unique(routes_df$layer_ix))
	n_experts = length(unique(routes_df$expert))
	
	question_df =
		routes_df %>%
		filter(., topk_ix == 1) %>%
		select(., -topk_ix) %>%
		group_by(sample_ix) %>%
		summarize(layers = list(c(1:n_layers)), route = list(expert), .groups = 'drop') %>%
		inner_join(question_df_raw, ., join_by(sample_ix))
	
	question_df_t2 =
		routes_df %>%
		filter(., topk_ix == 2) %>%
		select(., -topk_ix) %>%
		group_by(sample_ix) %>%
		summarize(layers = list(c(1:n_layers)), route = list(expert), .groups = 'drop') %>%
		inner_join(question_df_raw, ., join_by(sample_ix))
	
	question_df <<- question_df
	question_df_t2 <<- question_df_t2
	routes_df <<- routes_df
	n_experts <<- n_experts
	n_layers <<- n_layers
})


# Data Checks  ---------------------------------------------

## Calculate Topk Distribution ---------------------------------------------
local({
	
	routes_df %>%
		head(., 1e7) %>%
		count(expert, topk_ix, layer_ix) %>%
		# mutate(., is_top1 = ifelse(topk_ix == 1, 1, 0)) %>%
		group_by(layer_ix, expert, topk_ix) %>%
		summarize(., n = sum(n), .groups = 'drop') %>%
		filter(., layer_ix %in% c(0, 1, 5, 10, 12, 15)) %>%
		ggplot() +
		geom_col(aes(x = expert, y = n, fill = as.factor(topk_ix)), position = 'stack') +
		geom_hline(yintercept = 8/64, color = 'red') +
		facet_grid(
			rows = vars(layer_ix),
			cols = vars(topk_ix)
			) +
		labs(title = 'Frequency of experts as topk = 1 relative to other topks', x = 'Expert ID', y = 'Routing Counts') 

})

## Check Coverage ---------------------------------------------------
local({
	
	#' Print how many distinct routes are needed to cover the sample, conditional on different path lengths
	#'
	#' @param sample_level_df The sample level dataframe containing a column `route` which is a list-column of routed expert IDs
	#' @param test_levels A nested list of vectors for all the levels you want to test 
	#'
	#' @examples \dontrun{
	#'   # Print route coverage of layers 1:3, layers 1-3-5
	#'   test_levels = list(c(1:3), c(1, 3, 5))
	#'   print_path_counts(sample_level_df, test_levels)
	#' }
	#'
	#' @returns T
	print_path_counts = function(sample_level_df, test_levels) {
		print(str_glue('Sample tokens: {big_format(nrow(sample_level_df))}'))

		walk(test_levels, function(test_levs) {
			
			test_combo = 
				sample_level_df %>%
				mutate(., route = map(route, \(x) x[test_levs])) %>%
				group_by(route) %>%
				summarize(n_tokens = n(), .groups = 'drop') %>%
				arrange(desc(n_tokens)) %>%
				mutate(., cumprop_toks = cumsum(n_tokens)/sum(n_tokens), cumprop_rtes = (1:n())/nrow(.))
			
			print(str_glue(
				'Test levels: {paste0(test_levs, collapse = ", ")}',
				'\n- Distinct routes: {big_format(nrow(test_combo))}',
				'\n- Distinct routes to cover 95%: {big_format(nrow(filter(test_combo, cumprop_toks < .95)))}',
				'\n- Distinct routes to cover 80%: {big_format(nrow(filter(test_combo, cumprop_toks < .8)))}'
			))
		})
	}
	
	# Test combinations of layers that return a reasonable number of paths
	print_path_counts(
		question_df,
		list(
			c(1:16), c(1:12), c(4:16),
			c(1:3), c(2:4), c(1, 5, 10),
			c(1:4), c(1, 3, 5, 7), c(1:5),
			c(2:6), c(4:8),
			c(1:7), c(2:8), c(3:9), c(1, 3, 5, 7, 9, 11, 13)
			)
		)
	
	
	#' Print expert continuity by layer
	#'
	#' @param route_level_df A route level df with columns `layer_ix`, `expert`, `sample_ix`
	#'
	#' @returns A layer_ix level dataframe
	get_ec_by_layer = function(route_level_df) {
		route_level_df %>%
			group_by(., sample_ix) %>%
			arrange(., sample_ix, layer_ix, .by_group = T)  %>%
			mutate(., is_cont = ifelse(expert == lead(expert, 1), 1, 0)) %>%
			filter(., !is.na(is_cont)) %>%
			group_by(., layer_ix) %>%
			summarize(., ec = sum(is_cont)/n(), .groups = 'drop')
	}
	
	get_ec_by_layer(head(routes_df, 1e7))
	
	
	#' Get coverage stats by layer combination
	#' 
	#' @param route_level_df A sample-level dataframe with columns `sample_ix`, `expert_id`, `layer_ix`
	#' @param n_layers The number of layers routed to
	#' @param n_experts The number of experts routed to
	#' @param .verbose If T, prints progress
	#' 
	#' @returns A dataframe at a (start_layer, end_layer) level with entropy metrics
	get_entropy_by_layer = function(route_level_df, n_layers, n_experts, .verbose = F) {
		
		possible_layer_combs =
			expand_grid(start_layer = 1:n_layers, end_layer = 1:n_layers) %>%
			filter(end_layer > start_layer)

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
	
	entropy_df = get_entropy_by_layer(filter(routes_df, sample_ix <= 1e5), n_layers, n_experts, F)
	
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
	
	print_path_counts <<- print_path_counts 
	get_entropy_by_layer <<- get_entropy_by_layer
})


# Get Clusters --------------------------------------------------------------

## Fixed Layer Indices --------------------------------------------------------------
local({

	get_path_clusters = function(sample_level_df, subset_layers, .min_n_per_cluster = 20) {	
		
		sample_level_df_segment = 
			sample_level_df %>%
			mutate(., expert_path = map(route, \(x) x[subset_layers]))
		
		route_counts =
			sample_level_df_segment %>%
			group_by(., expert_path) %>%
			summarize(
				.,
				n_samples = n(),
				output_samples = paste0(sample(predicted_texts, min(length(predicted_texts), 15)), collapse = ','),
				context_samples = paste0(sample(question, min(length(question), 15)), collapse = '\n'),
				domain_samples = paste0(sample(domain, min(length(domain), 15)), collapse = '\n'),
				domains = list(domain),
				.groups = 'drop'
			)
		
		route_counts_final = 
			route_counts %>%
			mutate(., layer_path = list(subset_layers)) %>%
			filter(., n_samples >= .min_n_per_cluster) %>%
			transmute(
				.,
				layer_path,
				expert_path,
				path_str = map2_chr(layer_path, expert_path, \(x, y) str_c(str_c('[l', x, '] ', y), collapse = ' -> ')),
				n_samples,
				output_samples,
				context_samples,
				domain_samples,
				domains
			)

		return(route_counts_final)
	}
	
	paths =
		get_path_clusters(question_df, c(1:15), 1) %>%
		filter(., n_samples >= 5) %>%
		group_by(path_str) %>%
		mutate(., path_ix = cur_group_id()) %>%
		ungroup()
	
	paths %>%
		filter(., n_samples >= 10) %>%
		unnest_longer(domains, values_to = 'domain') %>%
		group_by(path_str, domain) %>%
		summarize(., n_count = n(), .groups = 'drop') %>%
		complete(path_str, domain, fill = list(n_count = 0)) %>% 
		left_join(select(paths, path_str, n_samples), by = 'path_str') %>%
		group_by(., desc(n_samples), path_str) %>%
		mutate(
			.,
			pct_domain = n_count/sum(n_count),
			path_ix = paste0('Path ', str_pad(cur_group_id(), pad = '0', width = 2), ' (n = ', n_samples, ')')
			) %>%
		ungroup() %>%
		ggplot() +
		geom_tile(aes(x = domain, y = path_ix, fill = (pct_domain) ^ .8)) +
		geom_text(aes(x = domain, y = path_ix, label = paste0(round(pct_domain * 100, 0), '%'))) +
		scale_fill_distiller(palette = 'YlGn', direction = 1) +
		scale_y_discrete(limits = rev) +
		theme_minimal() +
		labs(x = 'Domain', y = 'Path')
	
	# Graph paths -> do they re-merge?
	paths %>%
		unnest_longer(expert_path, indices_to = 'layer_ix', values_to = 'expert') %>%
		group_by(., layer_ix) %>%
		mutate(
			.,
			prop = n_samples/sum(n_samples),
			linewidth = ifelse(prop^1.0 * 10 < 3.5, prop^1.0 * 10, 3.5),
			alpha = 0.3 + (prop * 0.7),
			) %>%
		ungroup() %>%
		mutate(., expert = expert + .1 * path_ix) %>%
		ggplot() + 
		geom_line(
			aes(x = layer_ix, y = expert, group = path_ix, color = as.factor(path_ix), linewidth = linewidth, alpha = alpha),
			) +
		scale_linewidth_identity() +
		scale_alpha_identity() +
		theme_minimal() +
		labs(
			title = 'Diverge - Remerge',
			x = 'Layer Index',
			y = 'Expert ID'
		)
	

	
	paths %>%
		sample_n(., 10000) %>%
		transmute(
			., 
			routing_path = path_str,
			n_samples,
			n_distinct_token_ids,
			token_samples,
			token_samples_with_context
			) %>%
		write_csv(., str_glue('{model_prefix}-paths-2-to-6.csv'))
})


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


## Graph of all paths most strongly associated with token :
paths3 %>% filter(., str_detect(path_str, '\\[l8\\] 25$')) %>% View()



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