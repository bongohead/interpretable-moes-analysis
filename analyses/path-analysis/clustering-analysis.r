# Setup ---------------------------------------------------------------
## Load Libs ---------------------------------------------------------------
library(uwot)
library(dbscan)
library(rnndescent)
library(slider)
library(tidyverse)

## Get Data ---------------------------------------------------------------
vocab_df =
	read_csv('analyses/path-analysis/moonlight-vocab.csv', trim_ws = F) %>%
	transmute(., token_id, token = display_form) %>%
	mutate(token = str_replace(token, coll('[NEWLINE]'), '_linebreak_')) %>%
	mutate(., token = str_replace(token, coll('âĢĻ'), '\'')) %>%
 	mutate(., token = str_replace(token, coll('âĢľ'), '\'')) %>%
	mutate(., token = str_replace(token, coll('âĢĵ'), '-')) %>%
	mutate(., token = str_replace(token, coll('Ċ'), ' ')) %>%
	mutate(., token = str_replace(token, coll('âĢĿ'), '"')) %>%
	mutate(., token = str_replace(token, coll('âĢĺ'), '"'))

token_df =
	read_csv('analyses/path-analysis/olmoe-c4-outputs.csv') %>%
	group_by(., batch_ix, sequence_ix) %>%
	mutate(., seq_id = cur_group_id(), is_start_of_seq = ifelse(token_ix == min(token_ix), 1, 0)) %>%
	ungroup() %>%
	group_by(., batch_ix, sequence_ix, token_ix) %>%
	mutate(., sample_ix = cur_group_id()) %>%
	ungroup()

routes_df =
	read_csv('analyses/path-analysis/olmoe-c4-routes-top1.csv') %>%
	inner_join(
		select(token_df, seq_id, sample_ix, batch_ix, sequence_ix, token_ix, is_start_of_seq),
		join_by(batch_ix, sequence_ix, token_ix)
		) %>%
	select(., -batch_ix, -sequence_ix, -token_ix, -weight)

## Helpers ---------------------------------------------------------------
big_format = scales::number_format(accuracy = 1e-2, scale_cut = scales::cut_short_scale())

# Initial Checks  ---------------------------------------------

## Calculate Topk Distribution ---------------------------------------------
local({
	
	# Only if topk > 1 loaded
	routes_df %>%
		count(expert, topk_ix, layer_ix) %>% 
		mutate(., is_top1 = ifelse(topk_ix == 1, 1, 0)) %>% 
		group_by(layer_ix, expert, is_top1) %>%
		summarize(., n = sum(n), .groups = 'drop') %>%
		pivot_wider(., id_cols = c(layer_ix, expert), names_from = is_top1, values_from = n, names_prefix = 'is_top1_') %>%
		mutate(., pct_top1 = is_top1_1/(is_top1_0 + is_top1_1)) %>%
		filter(., layer_ix %in% c(0, 1, 5, 10, 15)) %>%
		ggplot() + 
		geom_col(aes(x = expert, y = pct_top1)) +
		geom_hline(yintercept = 8/64, color = 'red') +
		facet_grid(rows = vars(layer_ix)) + 
		labs(title = 'Frequency of experts as topk = 1 relative to other topks', x = 'Expert ID', y = '% Top 1')

})


# Path Analysis --------------------------------------------------------------

## Fixed Layer Indices --------------------------------------------------------------
local({
	
	# Identify top clusters - for each combination (layer0_expert, layer1_expert, ...), how 
	# many unique samples (batch_ix x sequence_ix x token_ix) belong? 
	sample_level_df =
		routes_df %>%
		filter(., topk_ix == 1) %>%
		select(., -topk_ix) %>%
		group_by(seq_id, sample_ix, is_start_of_seq, token_id) %>%
		summarize(route = list(expert), .groups = 'drop') %>%
		left_join(., vocab_df, by = 'token_id') %>%
		group_by(seq_id) %>%
		arrange(sample_ix, .by_group = T) %>%
		mutate(
			.,
			tok_fmt = ifelse(is_start_of_seq == 1, paste0('<bos>', token), token),
			prev_tok = slide_index_chr(tok_fmt, sample_ix, \(x) paste0(x, collapse = ''), .before = 8, .after = -1),
			next_tok = slide_index_chr(tok_fmt, sample_ix, \(x) paste0(x, collapse = ''), .before = -1, .after = 4),
			token_context = ifelse(
				is_start_of_seq == 1,
				paste0(prev_tok, '<bos>[[', str_sub(tok_fmt, 6), ']]', next_tok),
				paste0(prev_tok, '[[', tok_fmt, ']]', next_tok)
				)
			) %>%
		ungroup() %>%
		select(-prev_tok, -next_tok, -is_start_of_seq)
	
	# Let's find combinations of routes that work well!
	test_levels = list(c(1:3), c(2:4), c(1, 5, 10), c(1:4), c(1, 3, 5, 7), c(1:5), c(2:6))
	
	walk(test_levels, function(test_levs) {
		
		test_combo = 
			sample_level_df %>%
			mutate(., route = map(route, \(x) x[test_levs])) %>%
			group_by(route) %>%
			summarize(n_tokens = n(), .groups = 'drop') %>%
			arrange(desc(n_tokens)) %>%
			mutate(., cumprop_toks = cumsum(n_tokens)/sum(n_tokens), cumprop_rtes = (1:n())/nrow(.))
		
		print(str_glue(
			'Test levels: {paste0(test_levs, collapse = ", ")}
			Sample tokens: {big_format(nrow(sample_level_df))}
			Distinct routes: {big_format(nrow(test_combo))}
			Distinct routes to cover 95%: {big_format(nrow(filter(test_combo, cumprop_toks < .95)))}
			Distinct routes to cover 80%: {big_format(nrow(filter(test_combo, cumprop_toks < .8)))}
		'))
		
	})


	routes_to_check = list(1:6, 2:8)
	
	path_clusters = map(routes_to_check, .progress = T, function(layers) {
	
		sample_level_df_segment = 
			sample_level_df %>%
			mutate(., expert_path = map(route, \(x) x[layers]))
		
		route_counts =
			sample_level_df_segment %>%
			group_by(., expert_path) %>%
			summarize(
				.,
				n_samples = n(),
				n_distinct_token_ids = n_distinct(token_id),
				token_samples = paste0(sample(token, min(length(token), 20)), collapse = ','),
				token_samples_with_context = paste0(sample(token_context, min(length(token_context), 10)), collapse = '\n'),
				.groups = 'drop'
			)
		
		# entropy_df =
		# 	sample_level_df_segment %>%
		# 	group_by(expert_path, token_id) %>%
		# 	summarize(count = n(), .groups = "drop") %>%
		# 	group_by(expert_path) %>%
		# 	summarize(
		# 		total = sum(count),
		# 		shannon_entropy = {
		# 			p <- count / total
		# 			-sum(p * log2(p))
		# 		},
		# 		.groups = "drop"
		# 	)	
		
		route_counts_final = 
			# inner_join(route_counts, entropy_df, by = 'expert_path') %>%
			route_counts %>%
			mutate(.,layer_path = list(layers)) %>%
			filter(., n_samples >= 10)
		
		return(route_counts_final)
	})
	
	# route_counts_1_4 =
	# 	sample_level_df %>%
	# 	mutate(., expert_path = map(route, \(x) x[c(1:4)])) %>%
	# 	group_by(., expert_path) %>%
	# 	summarize(
	# 		.,
	# 		n_samples = n(),
	# 		n_distinct_token_ids = n_distinct(token_id),
	# 		token_samples = paste0(sample(token, min(length(token), 10)), collapse = ','),
	# 		token_samples_with_context = paste0(sample(token_context, min(length(token_context), 10)), collapse = '\n'),
	# 		.groups = 'drop'
	# 	) %>%
	# 	filter(., n_samples >= 10)
	
	# view(route_counts)
})
	
	
## Token Level Analysis ----------------------------------------------------
local({
	
	sample_level_df %>% count(token) %>% arrange(desc(n)) %>% view()
	
	test_token_df =
		sample_level_df %>%
		sample_n(., 100000, replace = F) %>%
		# filter(., token == '-' & token_id == 14) %>%
		mutate(., layers = list(1:16))
	
	# Check how many different routes there are
	test_token_df_long =
		test_token_df %>%
		select(sample_ix, route) %>%
		unnest_longer(route, values_to = 'expert') %>%
		group_by(sample_ix) %>%
		mutate(., layer_ix = 1:n()) %>%
		ungroup()
	
	possible_layer_combs =
		expand_grid(start_layer = 1:16,end_layer = 1:16) %>%
		filter(end_layer > start_layer) 

	t0 = Sys.time()
	layer_combos =
		possible_layer_combs %>%
		rowwise() %>%
		mutate(
			metrics = {
				
				path_length = end_layer - start_layer + 1
				
				# Filter the data to [start_layer, end_layer]
				df_sub =
					test_token_df_long %>%
					filter(layer_ix >= start_layer, layer_ix <= end_layer)
				
				# Convert to a "route" per token (sample_ix)
				subroutes_df =
					test_token_df_long %>%
					filter(layer_ix >= start_layer, layer_ix <= end_layer) %>%
					group_by(sample_ix) %>%
					# arrange(layer_ix, .by_group = T) %>%
					summarize(route = list(expert), .groups = "drop") 
				
				subroute_counts =
					subroutes_df %>%
					group_by(route) %>%
					summarize(n_tokens = n(), .groups = "drop") %>%
					arrange(desc(n_tokens)) %>%
					mutate(
						.,
						cumprop_tokens = cumsum(n_tokens)/sum(n_tokens),
						cumprop_routes = (1:n())/nrow(.)
					)
				
				entropy =
					subroute_counts %>%
					mutate(prob = n_tokens / sum(n_tokens)) %>%
					summarize(
						obs_entropy = -sum(prob * log2(prob), na.rm = T),
						max_entropy = path_length * log2(64) # log2 = % of bits entropy
					) %>%
					mutate(., norm_entropy = obs_entropy/max_entropy)
				
				tibble(
					path_length = path_length,
					max_distinct_rtes = 64^path_length,
					n_distinct_rtes = nrow(subroute_counts),
					n_total_toks = sum(subroute_counts$n_tokens),
					t10_coverage = head(filter(subroute_counts, cumprop_routes > .1), 1)$cumprop_tokens,
					t50_coverage = head(filter(subroute_counts, cumprop_routes > .5), 1)$cumprop_tokens,
					obs_entropy = entropy$obs_entropy,
					max_entropy = entropy$max_entropy,
					norm_entropy = entropy$norm_entropy # Fraction of bits of capacity actually used
					# pct_rcts_to_cap_50pct_toks = nrow(filter(subroute_counts, cumprop_tokens < .5))/nrow(subroute_counts),
					# pct_rcts_to_cap_90pct_toks = nrow(filter(subroute_counts, cumprop_tokens < .9))/nrow(subroute_counts)
				)
			}
		) %>%
		ungroup() %>%
		unnest(cols = metrics)
	Sys.time() - t0
	
	layer_combos %>%
		ggplot() +
		geom_tile(aes(x = start_layer, y = path_length, fill = t10_coverage)) +
		shadowtext::geom_shadowtext(
			aes(x = start_layer, y = path_length, label = scales::label_percent(accuracy = 1)(t10_coverage)),
			size = 3
			) +
		scale_x_continuous(breaks = 1:16) +
		scale_fill_viridis_c(direction = 1) +
		theme_minimal() +
		labs(title = 'Token coverage from top 10% of routes', x = 'Starting Layer', y = 'Path Length') +
		theme(legend.position = 'none')
	
	layer_combos %>%
		ggplot() +
		geom_tile(aes(x = start_layer, y = path_length, fill = norm_entropy)) +
		shadowtext::geom_shadowtext(
			aes(x = start_layer, y = path_length, label = scales::label_percent(accuracy = 1)(norm_entropy)),
			size = 3
		) +
		scale_x_continuous(breaks = 1:16) +
		scale_fill_viridis_c(direction = 1) +
		theme_minimal() +
		labs(title = 'Normalized entropy (fraction of bits of capacity used)', x = 'Starting Layer', y = 'Path Length') +
		theme(legend.position = 'none')
	
	test_token_routes = list(1:4, 2:6, 4:8, 2:12)
	
	test_token_path_clusters = map(test_token_routes, .progress = T, function(layers) {
		test_token_df %>%
			mutate(., expert_path = map(route, \(x) x[layers])) %>%
			group_by(., expert_path) %>%
			summarize(
				.,
				n_samples = n(),
				n_distinct_token_ids = n_distinct(token_id),
				token_samples = paste0(sample(token, min(length(token), 20)), collapse = ','),
				token_samples_with_context = paste0(sample(token_context, min(length(token_context), 10)), collapse = '\n'),
				.groups = 'drop'
			) %>%
			mutate(.,layer_path = list(layers)) %>%
			filter(., n_samples >= 5)
	})
	
	
	

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


# Cluster Based Path Analysis ---------------------------------------------
test_token_df =
	sample_level_df %>%
	# sample_n(., 500000, replace = F) %>%
	mutate(., layers = list(1:16)) %>%
	mutate(., cl_route = map(route, \(x) x[1:10]))

test_token_df_wide = 
	test_token_df %>%
	select(sample_ix, cl_route) %>%
	unnest_wider(., cl_route, names_sep = '_', names_repair = \(x) str_replace_all(x, 'cl_route_', 'layer_'))

# Clustering sample
cluster_sample_df = sample_n(test_token_df_wide, 100000)
cluster_sample_mat = as.matrix(select(cluster_sample_df, starts_with('layer_')))

umap_emb = umap(
	X = cluster_sample_mat,
	metric = 'hamming',
	nn_method = 'nndescent', # approximate nearest neighbors
	n_neighbors = 10, # Smaller more nuanced neighborhood size
	n_components = 2,
	# min_dist = 0.1, # tweak for more/less cluster separation
	verbose = T,
	ret_model = T,
	seed = 123
)

# kNNdistplot(umap_emb$embedding, k = 4)
# dbscan_res = dbscan(umap_emb$embedding, eps = 1, minPts = 10)
hdbscan_res = dbscan(umap_emb$embedding, eps = 1, minPts = 10)

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
	mutate(., cluster_id = dbscan_res$cluster) %>%
	ggplot() +
	geom_point(aes(x = emb_1, y = emb_2, color = as.factor(cluster_id)))


# # Route-level plot
# cluster_sample_mat_clustered %>%
# 	mutate(., short_route = map(route, \(x) x[1:10])) %>%
# 	group_by(short_route, cluster_id)
# 

# kmodes_obj = kmeans(umap_emb$embedding, centers = 100, iter.max = 1000)
# # smaller => more, smaller clusters; bigger => fewer, bigger clusters
# # hdb = hdbscan(x = umap_emb$embedding,	minPts = 10)
# hdb = hdbscan(x = fr,	minPts = 10)
# 
# cluster_sample_mat_clustered =
# 	cluster_sample_df %>%
# 	mutate(., cluster_id = hdb$cluster)
# 
# df_cluster_sum <- cluster_assignments %>%
# 	group_by(cluster_id) %>%
# 	summarize(count = n()) %>%
# 	arrange(desc(count))
# 
# viz_df <- data.frame(
# 	UMAP1      = umap_emb[,1],
# 	UMAP2      = umap_emb[,2],
# 	cluster_id = factor(hdb$cluster)
# )
# 
# ggplot(viz_df, aes(x=UMAP1, y=UMAP2, color=cluster_id)) +
# 	geom_point(size=0.5, alpha=0.6) +
# 	theme_minimal() +
# 	scale_color_discrete(na.translate = FALSE) +
# 	labs(title="HDBSCAN Clusters of 50k Sample in UMAP-Hamming Space")
# 
# 
# as_tibble(umap_emb$embedding)
