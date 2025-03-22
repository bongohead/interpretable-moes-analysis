# Setup ---------------------------------------------------------------
## Load Libs ---------------------------------------------------------------
library(tidyverse)

## Get Data ---------------------------------------------------------------
routes_df =
	read_csv('analyses/cluster-analysis/olmoe-c4-routes-top1.csv') %>%
	group_by(., batch_ix, sequence_ix) %>%
	mutate(., is_start_of_seq = ifelse(token_ix == min(token_ix), 1, 0)) %>%
	ungroup() %>%
	group_by(., batch_ix, sequence_ix, token_ix, token_id) %>%
	mutate(., sample_ix = cur_group_id()) %>%
	ungroup() %>%
	select(., -batch_ix, -sequence_ix, -token_ix, -weight)

vocab_df =
	read_csv('analyses/cluster-analysis/olmoe-vocab.csv', trim_ws = F) %>%
	transmute(., token_id, token = display_form)


# Calculate Topk Distribution ---------------------------------------------
local({
	
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

# Clustering --------------------------------------------------------------
local({
	
	# Identify top clusters - for each combination 
	# [e0, e1, e2, e3], how many unique samples (batch_ix x sequence_ix x token_ix) belong? 
	# routes_df %>%
	# 	filter(., topk_ix == 1 & layer_ix %in% 0:4) %>%
	# 	select(., -topk_ix, -token_id) %>%
	# 	pivot_wider(., id_cols = c(sample_ix), names_from = layer_ix, values_from = expert, names_prefix = 'expert_') %>%
	# 	group_by(., expert_0, expert_1, expert_2, expert_3, expert_4) %>%
	# 	summarize(., n = n(), .groups = 'drop') %>%
	# 	arrange(desc(n)) %>%
	# 	mutate(., cumprop = cumsum(n)/sum(n)) %>%
	# 	View()
	
	pad = \(x) str_pad(x, 2, 'left', '0')

	sample_level_df =
		routes_df %>%
		filter(., topk_ix == 1) %>%
		select(., -topk_ix) %>%
		group_by(sample_ix, is_start_of_seq, token_id) %>%
		summarize(
			route = list(expert), # paste0(pad(expert), collapse = "-"),
			.groups = "drop"
		) %>%
		left_join(., vocab_df, by = 'token_id') %>%
		arrange(sample_ix) %>%
		mutate(
			.,
			tok_fmt = ifelse(is_start_of_seq == 1, paste0('<eos>', token), token),
			prev_tok = slide_index_chr(tok_fmt, sample_ix, \(x) paste0(x, collapse = ''), .before = 6, .after = -1),
			next_tok = slide_index_chr(tok_fmt, sample_ix, \(x) paste0(x, collapse = ''), .before = -1, .after = 6),
			token_context = paste0(prev_tok, '[[', tok_fmt, ']] ', next_tok)
			) %>%
		select(-prev_tok, -next_tok, -is_start_of_seq)
	
	route_counts =
		sample_level_df %>%
		mutate(., expert_path = map(route, \(x) x[c(2:8)])) %>%
		group_by(., expert_path) %>%
		summarize(
			.,
			n_samples = n(),
			n_distinct_token_ids = n_distinct(token_id),
			token_samples = paste0(sample(token, min(length(token), 10)), collapse = ','),
			token_samples_with_context = paste0(sample(token_context, min(length(token_context), 10)), collapse = '\n'),
			.groups = 'drop'
		) %>%
		filter(., n_samples >= 10)
	
	view(route_counts)

	# sample_level_df %>% mutate(., route_trunc = map(route, \(x) x[1:3]))

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
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
	
	
