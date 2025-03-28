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

## Get Data ---------------------------------------------------------------
local({
	
	question_df_raw =
		read_csv(str_glue('analyses/mmlu-activations-study/data/{model_prefix}-mmlu-questions.csv')) %>%
		rename(., sample_ix = q_ix)

	routes_df =
		read_csv(str_glue('analyses/mmlu-activations-study/data/{model_prefix}-mmlu-topk.csv')) %>%
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
	
	
	print_path_counts <<- print_path_counts 
})

routes_df %>% filter(., layer_ix == 15) %>% count(expert) 


# Analysis --------------------------------------------------------------


## Overall Routing Coutns --------------------------------------------------
local({
	
	# Routing counts (barplot) by expert ID and domain
	question_df %>%
		unnest_longer(route, values_to = 'expert', indices_to = 'layer_ix') %>% 
		mutate(domain = ifelse(domain == 'math', 'math', 'not_math')) %>% 
		group_by(., layer_ix, expert, domain) %>% 
		summarize(., n = n(), .groups = 'drop') %>% 
		ggplot() + geom_bar(aes(x = expert, y = n, group = domain, fill = domain), stat = 'identity') +
		facet_wrap(vars(layer_ix), scales = 'free_y') +
		theme_minimal() +
		labs(x = 'Expert ID', y = 'Count', title = 'Count by Expert ID & Domain')
	
	# Routing counts (lineplot) by expert ID and domain
	question_df %>%
		unnest_longer(route, values_to = 'expert', indices_to = 'layer_ix') %>% 
		mutate(d2 = ifelse(domain == 'math', 'math', 'other')) %>%
		filter(., sample_ix %in% c(
			sample(unique(filter(., !domain %in% c('math'))$sample_ix), 50),
			sample(unique(filter(., domain == 'math')$sample_ix), 50)
		)) %>%
		mutate(., expert = expert + rnorm(nrow(.), 0, 1)) %>%
		ggplot() + 
		geom_line(
			aes(x = layer_ix, y = expert, group = sample_ix, color = as.factor(d2)), 
			linewidth = 1.5,
			alpha = .1
		) +
		theme_minimal() +
		labs(x = 'Layer Index', y = 'Expert ID', color = 'Domain', title = 'Routing Path by Domain')
	
	## Counts by Answer
	question_df %>%
		unnest_longer(route, values_to = 'expert', indices_to = 'layer_ix') %>% 
		group_by(., layer_ix, expert, predicted_choice) %>% 
		summarize(., n = n(), .groups = 'drop') %>% 
		ggplot() + geom_bar(aes(x = expert, y = n, group = predicted_choice, fill = predicted_choice), stat = 'identity') +
		facet_wrap(vars(layer_ix), scales = 'free_y') +
		theme_minimal() +
		labs(x = 'Expert ID', y = 'Count', title = 'Count by Expert ID & Answer')
	
	## Counts by Correctness
	question_df %>%
		unnest_longer(route, values_to = 'expert', indices_to = 'layer_ix') %>% 
		group_by(., layer_ix, expert, is_correct) %>% 
		summarize(., n = n(), .groups = 'drop') %>% 
		ggplot() + geom_bar(aes(x = expert, y = n, group = is_correct, fill = as.factor(is_correct)), stat = 'identity') +
		facet_wrap(vars(layer_ix), scales = 'free_y') +
		theme_minimal() +
		labs(x = 'Expert ID', y = 'Count', title = 'Count by Expert ID & Correctness')
	
	
})


## Accuracy by Domain --------------------------------------------------------------
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
	

	# Define paths
	paths =
		get_path_clusters(question_df, c(1:10), 1) %>%
		filter(., n_samples >= 5) %>%
		group_by(path_str) %>%
		mutate(., path_ix = cur_group_id()) %>%
		ungroup()
	
	paths %>%
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
		labs(title = 'Domain Coverage by Path', x = 'Domain', y = 'Path') +
		theme(legend.position = 'none')
	
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
	
	# Just look at math paths only
	math_paths =
		get_path_clusters(question_df %>% filter(domain == 'math'), c(1:12), 1) %>%
		filter(., n_samples >= 3) %>%
		group_by(path_str) %>%
		mutate(., path_ix = cur_group_id()) %>%
		ungroup()
	
	math_paths %>%
		unnest_longer(expert_path, indices_to = 'layer_ix', values_to = 'expert') %>%
		group_by(., layer_ix) %>%
		mutate(
			.,
			prop = n_samples/sum(n_samples),
			linewidth = ifelse(prop^1.0 * 10 < .5, prop^1.0 * 100, .5),
			alpha = 0.3 + (prop * 0.7),
		) %>%
		ungroup() %>%
		mutate(., expert = expert + .2 * path_ix) %>%
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
	
})


## Auto Clusters ---------------------------------------------
local({
	
	grouping_layers = 1:16
	cl_sample_size = 20000
	
	test_token_df =
		question_df %>%
		# filter(domain == 'math') %>%
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
		n_neighbors = 30, # Smaller more nuanced neighborhood size
		n_components = 2, # 2-dimensional
		# min_dist = 0.1, # tweak for more/less cluster separation
		verbose = T,
		ret_model = T,
		seed = 123
	)
	
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
	cl_samples =
		cluster_sample_mat_clustered %>%
		as_tibble() %>%
		mutate(., cluster_id = hdbscan_res$cluster, cl2 = ifelse(domain == 'math', 'math', 'other'))
	
	cl_samples %>%
		filter(domain %in% c('math', 'cs', 'biology', 'chemistry', 'statistics')) %>%
		mutate(., emb_1 = emb_1 + rnorm(nrow(.), 0, .5), emb_2 = emb_2 + rnorm(nrow(.), 0, .5)) %>%
		ggplot() +
		geom_point(
			aes(x = emb_1, y = emb_2, color = as.factor(domain)),
			alpha = .75
			) +
		theme_minimal() +
		labs(
			labs = 'UMAP 2-dimensional path clusters for token "of"',
			color = 'Domain',
			x = 'Dimension 1',
			y = 'Dimension 2'
			)
	
	cl_samples %>%
		mutate(
			., 
			layers = map(layers, \(x) x[grouping_layers]),
			path_str = map2_chr(layers, route, \(x, y) str_c(str_c('[l', x, '] ', y), collapse = ' -> '))
			) %>%
		group_by(cluster_id) %>%
		summarize(
			.,
			n_samples = n(),
			output_samples = paste0(sample(predicted_texts, min(length(predicted_texts), 15)), collapse = ','),
			context_samples = paste0(sample(question, min(length(question), 15)), collapse = '\n'),
			domain_samples = paste0(sample(domain, min(length(domain), 15)), collapse = '\n'),
			domains = list(domain),
			.groups = 'drop'
		)
	
	cl_samples %>%
		head(., 10) %>%
		unnest_longer(route, indices_to = 'layer_ix', values_to = 'expert') %>%
		mutate(., expert = expert + rnorm(nrow(.), 0, .5)) %>%
		ggplot() + 
		geom_line(
			aes(x = layer_ix, y = expert, group = as.factor(sample_ix), color = as.factor(cluster_id)),
			linewidth = 0.5,
			alpha = 0.2
		) +
		scale_linewidth_identity() +
		scale_alpha_identity() +
		theme_minimal() +
		labs(
			title = 'Diverge - Remerge',
			x = 'Layer Index',
			y = 'Expert ID'
		)
	
	
})

# Calculate Jensen Shannon Divergence -------------------------------------

## Domain JSD -------------------------------------
local({
	
	#' Get JS distance
	#' 
	#' @param p A vector of distribution 1
	#' @param q A vector of distribution 2, the same size as `p`
	get_js_distance = function(p, q) {
		# small epsilon to avoid log(0)
		eps = 1e-12
		
		# Ensure p and q sum to 1 (in case of minor floating error)
		p = p / sum(p + eps)
		q = q / sum(q + eps)
		
		m = 0.5 * (p + q)
		
		# KL(p || m) in base 2
		kl_pm = sum(p * (log2((p + eps) / (m + eps))))
		# KL(q || m) in base 2
		kl_qm = sum(q * (log2((q + eps) / (m + eps))))
		
		# JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
		js = 0.5 * kl_pm + 0.5 * kl_qm
		js_dist = sqrt(js)
		
		return(js_dist)
	}

	#' Get JS distance metrics by layer
	#'
	#' @param input_df A df with cols `layer_ix`, `groupcol`, `expert`, `n` giving counts at each 
	#'  `layer_ix` x `groupcol` `expert`. 0s should be filled in if not already so.
	#' @param grouping_col The name of the grouping column.
	#' 
	#' @examples \dontrun{
	#' input_df = 
	#'   routes_df %>%
	#' 	 left_join(select(question_df, sample_ix, domain), by = 'sample_ix') %>%
	#' 	 filter(., topk_ix == 1) %>% # Optional
	#' 	 select(., -weight, -topk_ix, -sample_ix) %>%
	#' 	 group_by(., layer_ix, domain, expert) %>%
	#' 	 summarize(., n = n(), .groups = 'drop') %>%
	#' 	 left_join(
	#' 		  expand_grid(
	#' 			 layer_ix = sort(unique(.$layer_ix)),
	#' 			 domain = sort(unique(.$domain)),
	#' 			 expert = 1:n_experts
	#' 		  ),
	#' 		  .,
	#' 		  join_by(layer_ix, domain, expert)
	#' 	 ) %>%
	#' 	 mutate(., n = replace_na(n, 0))
	#' 	 
	#' get_js_by_layer(input_df, 'domain')
	#' }
	#' 
	#' @returns A dataframe at a layer_ix level.
	get_js_by_layer = function(input_df, grouping_col) {
		
		df =
			input_df %>%
			rename(segment = !!grouping_col)
		
		p_by_seg =
			df %>%
			group_by(., layer_ix, segment) %>%
			mutate(., p_l_s_e = n/sum(n)) %>%
			ungroup()
		
		p_by_all = 
			df %>%
			group_by(., layer_ix, expert) %>%
			summarize(., n = sum(n), .groups = 'drop') %>%
			group_by(., layer_ix) %>%
			mutate(., p_l_e = n/sum(n)) %>%
			ungroup()
		
		p_by_seg %>% inner_join(p_by_all, by = c('layer_ix', 'expert'))
		
		p_by_seg %>%
			inner_join(p_by_all, by = c('layer_ix', 'expert')) %>%
			group_by(., layer_ix, segment) %>%
			summarize(
				.,
				p_l_s_e = list(p_l_s_e),
				p_l_e = list(p_l_e),
				.groups = 'drop'
			) %>%
			mutate(
				js_value = mapply(function(x, y) js_distance(x, y), p_l_s_e, p_l_e)
			) %>%
			group_by(., layer_ix) %>%
			summarize(., js_l = mean(js_value), .groups = 'drop') 
	}
	
	input_df =
	  routes_df %>%
		 left_join(select(question_df, sample_ix, domain), by = 'sample_ix') %>%
		 select(., -weight, -topk_ix, -sample_ix) %>%
		 group_by(., layer_ix, domain, expert) %>%
		 summarize(., n = n(), .groups = 'drop') %>%
		 left_join(
			  expand_grid(
				 layer_ix = sort(unique(.$layer_ix)),
				 domain = sort(unique(.$domain)),
				 expert = 1:n_experts
			  ),
			  .,
			  join_by(layer_ix, domain, expert)
		 ) %>%
		 mutate(., n = replace_na(n, 0))
	
	get_js_by_layer(input_df, 'domain') %>%
		ggplot() +
		geom_line(aes(x = layer_ix, y = js_l))
	
})