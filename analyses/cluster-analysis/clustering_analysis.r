# Setup ---------------------------------------------------------------
## Load Libs ---------------------------------------------------------------
library(tidyverse)
library(highcharter)
cluster_rate = 4

## Get Data ---------------------------------------------------------------
raw_routing =
	read_csv('analyses/cluster-analysis/olmoe_clustering.csv')

n_layers = length(unique(raw_routing$layer_ix))
n_experts = 64/cluster_rate
pad = \(x) str_pad(x, 2, 'left', '0')

raw_routing %>%
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
	facet_grid(rows = vars(layer_ix))


raw_routing %>%
	filter(., topk_ix == 1) %>%
	filter(., )
