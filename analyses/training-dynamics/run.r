library(tidyverse)
library(bit64)
# Bash: scp -i id_rsa_runpod -P 47343 root@195.26.232.156:/workspace/interpretable-moes/experiments/base-olmoe-lflb/logs/*.csv /data
# PS: scp -P 47343 root@195.26.232.156:"/workspace/interpretable-moes/experiments/base-olmoe-lflb/logs/*.csv" data


# Get Data ----------------------------------------------------------------
files = sort(fs::dir_ls('data'))
raw_data = list_rbind(imap(files, function(i, fname) {
	
	total_tokens_trained =
		fname %>%
		basename() %>%
		tools::file_path_sans_ext() %>%
		str_replace(., 's_', '') %>%
		as.integer64()
	
	raw_csv = read_csv(fname, col_types = cols(.default = col_integer()))
	
	res =
		raw_csv %>%
		head(., 5e6) %>%
		group_by(., batch_ix, seq_ix, token_ix) %>%
		mutate(., sample_ix = cur_group_id() - 1) %>%
		ungroup() %>%
		select(., -batch_ix, -seq_ix, -token_ix) %>%
		mutate(., train_ix = total_tokens_trained)
	# filter(., topk_slot == 0) %>%
	
	gc()
	return(res)
}))

token_map = transmute(read_csv('vocab.csv'), token_id, token = display_form)


# Base Stats --------------------------------------------------------------
top_token_ids =
	raw_data %>%
	filter(train_ix == min(train_ix) & topk_slot == 0 & layer_ix == 0) %>% 
	count(token_id) %>%
	arrange(desc(n)) %>%
	head(., 1000) %>%
	inner_join(token_map, by = 'token_id')

# 2656 = God
# 4370 = dog
# 5754 = prison
train_steps = count(raw_data, train_ix)

# Visualizations ----------------------------------------------------------
counts_by_layer_expert_train =
	raw_data %>% 
	filter(., token_id == 5754) %>%
	filter(., train_ix %in% c(524288, 105379948, 1049078433, 2097632210, 5243291559)) %>%
	group_by(., layer_ix, expert_id, train_ix) %>%
	summarize(., n = n(), .groups = 'drop') %>%
	group_by(., layer_ix, train_ix) %>%
	mutate(., prop = n/sum(n)) %>%
	ungroup() %>%
	right_join(
		.,
		expand_grid(layer_ix = 0:13, expert_id = 0:47, train_ix = unique(.$train_ix)),
		by = c('layer_ix', 'expert_id', 'train_ix')
		) %>%
	mutate(., n = replace_na(n, 0), prop = replace_na(prop, 0))


counts_by_layer_expert_train %>% 
	filter(., layer_ix == 1) %>%
	mutate(
		# Create a factor with levels in the correct order
		train_ix_ordered = factor(train_ix, levels = c(524288, 105379948, 1049078433, 2097632210)),
		# Add the formatted label for display
		train_ix_formatted = case_when(
			train_ix < 1e6 ~ sprintf("%.2fK", train_ix/1e3),
			train_ix < 1e9 ~ sprintf("%.2fM", train_ix/1e6),
			TRUE ~ sprintf("%.2fB", train_ix/1e9)
		),
		train_ix_formatted = paste0(train_ix_formatted, ' tokens trained'),
		facet_label = factor(train_ix_formatted,levels = unique(train_ix_formatted[order(train_ix_ordered)]))
	) %>%
	ggplot(., aes(expert_id, prop)) +
	geom_col(
		width = 0.9,  # Slightly narrower bars
		fill = "steelblue",
		color = "#4682B4",  # Add matching border color
		alpha = 0.9,  # Slight transparency
		linewidth = 0.1
	) +
	coord_radial(
		inner.radius = 0.1,
		rotate_angle = F,
		expand = F
	) +
	scale_y_continuous(
		trans = scales::trans_new( # lower = more extreme
			name = paste0("power_", .3),
			transform = function(x) x^.3,
			inverse = function(x) x^(1/.3)
		),
		limits = c(0, 1),
		labels = NULL,
		breaks = c(0, 1)
		) +
	scale_x_continuous(
		breaks = function(x) seq(floor(x[1]), ceiling(x[2]), by = 1),
		labels = NULL
		# labels = function(x) ifelse(x %in% c(0, 10, 20, 30, 40), as.character(x), "")
	) +
	theme(
		axis.text.theta = element_text(angle = 0, hjust = 0.5),  # Better angle for readability
		legend.position = "none",  # Remove legends
		panel.grid.major = element_line(color = "gray90"),
		panel.grid.minor = element_line(color = "gray95", linewidth = 0.3)
	) +
	facet_wrap(vars(facet_label), nrow = 1) +
	labs(x = NULL, y = NULL) +
	ggthemes::theme_fivethirtyeight()





















raw_data %>% 
	filter(., token_id == 253 & topk_slot == 0) %>%
	group_by(train_ix, expert_id) %>%
	summarize(., n = n(), .groups = 'drop') %>%
	ggplot() +
	scale_x_continuous(
		labels = scales::label_number(scale = 1e-6, suffix = "M"),
		name = "Value (millions)"
	) +
	geom_line(aes(x = train_ix, y = n, color = as.factor(expert_id), group = expert_id))

# For a fixed B, token idx, what is the consistency
raw_data %>%
	filter(., token_id == 253 & topk_slot == 0 & layer_ix == 0) %>%
	group_by(., sample_ix) %>%
	arrange(., sample_ix, total_tokens_trained) %>%
	mutate(., expert_cont = ifelse(expert_id == lag(expert_id, 1), 1, 0)) %>%
	ungroup() %>%
	na.omit() %>%
	count(total_tokens_trained, expert_cont) %>%
	ggplot() + 
	geom_line(aes(x = total_tokens_trained, y = n, color = as.factor(expert_cont), group = as.factor(expert_cont)))


raw_data %>%
	filter(., topk_slot == 0 & total_tokens_trained >= 52952624) %>%
	group_by(., sample_ix, layer_ix) %>%
	arrange(., sample_ix, layer_ix, total_tokens_trained) %>%
	mutate(., is_expert_same = ifelse(expert_id == head(expert_id, 1), 1, 0)) %>%
	ungroup() %>%
	group_by(., layer_ix, total_tokens_trained) %>%
	summarize(., pct_recycled_same = sum(is_expert_same)/n(), .groups = 'drop') %>%
	mutate(., layer_ix = as.factor(layer_ix)) %>%
	ggplot() + 
	geom_line(aes(x = total_tokens_trained, y = pct_recycled_same, color = layer_ix)) +
	geom_point(aes(x = total_tokens_trained, y = pct_recycled_same, color = layer_ix))


raw_data %>%
	filter(., topk_slot == 0 & total_tokens_trained >= train_steps$total_tokens_trained[[3]]) %>%
	group_by(., sample_ix, layer_ix) %>%
	arrange(., sample_ix, layer_ix, total_tokens_trained) %>%
	mutate(., is_expert_same = ifelse(expert_id == head(expert_id, 1), 1, 0)) %>%
	ungroup() %>%
	group_by(., layer_ix, total_tokens_trained) %>%
	summarize(., pct_recycled_same = sum(is_expert_same)/n(), .groups = 'drop') %>%
	mutate(., layer_ix = as.factor(layer_ix)) %>%
	ggplot() + 
	geom_line(aes(x = total_tokens_trained, y = pct_recycled_same, color = layer_ix)) +
	geom_point(aes(x = total_tokens_trained, y = pct_recycled_same, color = layer_ix)) +
	ggthemes::theme_base() +
	scale_x_continuous(
		labels = scales::label_number(scale = 1e-9, suffix = 'B')
	) +
	labs(
		x = 'Total Tokens Trained',
		y = 'Pct Retaining Same Experts'
	)

# Temp
raw_data %>%
	filter(., token_id == 13 & topk_slot == 0 & layer_ix == 1) %>%
	group_by(., expert_id, total_tokens_trained, layer_ix) %>%
	summarize(., n = n(), .groups = 'drop') %>%
	group_by(., total_tokens_trained, layer_ix) %>%
	mutate(., pct_of_total = n/sum(n)) %>%
	ungroup() %>%
	# filter(total_tokens_trained == min(total_tokens_trained)) %>%
	ggplot() +
	geom_col(aes(x = expert_id, y = pct_of_total)) +
	facet_wrap(vars(total_tokens_trained))



raw_data %>%
	filter(., batch == 0 & token_idx == 0 & layer_idx == 0 & topk_slot == 0 & total_tokens_trained == max(total_tokens_trained)) 