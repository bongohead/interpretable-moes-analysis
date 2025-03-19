# Setup ---------------------------------------------------------------
## Load Libs ---------------------------------------------------------------
library(tidyverse)
library(highcharter)
cluster_rate = 4

## Get Data ---------------------------------------------------------------
raw_routing =
	read_csv('analyses/cross-layer-routing/contextual_token_activations.csv') %>%
	filter(is_test_token == 1)

n_layers = length(unique(raw_routing$layer_ix))
n_experts = 64/cluster_rate
pad = \(x) str_pad(x, 2, 'left', '0')


# Analysis ----------------------------------------------------------------

## Sankey Chart ----------------------------------------------------------------
local({
	
	# Get token x meaning x sample x layer level routes, top-1 only
	sample_x_layer =
		raw_routing %>%
		rename(expert_id = expert) %>%
		filter(topk_ix %in% 1:1) %>%
		group_by(test_token, meaning_label, batch_ix, sequence_ix, token_ix) %>%
		mutate(., sample_ix = cur_group_id()) %>%
		ungroup() %>%
		select(., token, meaning_label, sample_ix, layer_ix, expert_id) %>%
		mutate(., expert_id = floor(expert_id / cluster_rate))
	
	# At each token x meaning x source layer x dest layer, which have the any given (source expert, dest expert)
	from_layer_x_to_layer = 
		inner_join(
			sample_x_layer %>%
				transmute(., token, meaning_label, sample_ix, from_layer = layer_ix, from_expert = expert_id),
			sample_x_layer %>%
				transmute(., token, meaning_label, sample_ix, to_layer = layer_ix, to_expert = expert_id, from_layer = layer_ix - 1),
			join_by(token, meaning_label, sample_ix, from_layer)
		) %>%
		group_by(token, meaning_label, from_layer, to_layer, from_expert, to_expert) %>%
		summarize(., count = n(), .groups = 'drop') %>%
		group_by(token, meaning_label, from_layer, to_layer) %>%
		mutate(., prop = count/sum(count)) %>%
		ungroup()
	
	# Join in empty data
	from_layer_x_to_layer_all =
		inner_join(
			crossing(from_layer = 0:(n_layers - 2), from_expert = 0:(n_experts - 1)),
			crossing(to_layer = 1:(n_layers - 1), to_expert = 0:(n_experts - 1)) %>%
				mutate(., from_layer = to_layer - 1),
			join_by(from_layer),
			relationship = 'many-to-many'
		) %>%
		cross_join(distinct(from_layer_x_to_layer, token, meaning_label), .) %>%
		left_join(
			from_layer_x_to_layer,
			join_by(token, meaning_label, from_layer, to_layer, from_expert, to_expert)
			) %>%
		mutate(
			from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
			to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
			count = replace_na(count, 0),
			prop = replace_na(prop, 0)
		)
	
	
	all_charts = map(group_split(from_layer_x_to_layer_all, token), function(samples_for_token) {
		token = samples_for_token$token[[1]]
		print(token)
		
		hc_charts =	map(group_split(samples_for_token, meaning_label), function(samples_for_meaning) {
			
			sample_count = samples_for_meaning %>% filter(str_detect(from, 'L00')) %>% .$count %>% sum()
			meaning_label = samples_for_meaning$meaning_label[[1]]
			
			nodes =
				crossing(layer = 0:(n_layers - 1), expert = 0:(n_experts - 1)) %>%
				transmute(
					id = paste0("L", pad(layer), "_E", pad(expert)),
					name = paste0("L", pad(layer), "_E", pad(expert)),
					height = 10,
					offsetVertical = 0,
					offsetHorizontal = 0,
					column = expert,
					level = layer,
					color = viridisLite::viridis(n_layers)[layer + 1]
				) %>%
				purrr::transpose()
			
			chart_data =
				samples_for_meaning %>%
				transmute(
					.,
					from, 
					to,
					color = ifelse(prop == 0, 'transparent', NA),
					weight = ifelse(prop == 0, .001, prop/100),
				)
			
			hc = 
				highchart() %>%
				hc_chart(
					height = 500,
					type = "sankey",
					marginLeft = 40,  # Add left margin for y-axis label
					marginBottom = 40 # Add bottom margin for x-axis label
				) %>%
				hc_title(
					text = str_glue(
						# '<h4>Expert routing</h4>',
						'<br><span style="font-size:14px;">',
						'Routing for token <span style="color:deeppink; font-weight:bold; font-style:italic">{token}</span> ',
						'used in the context of <span style="color: deeppink">{meaning_label}</span>',
						'<span style="font-size:12px;"> ({sample_count} samples)</span>',
						'</span>'
						)
				) %>%
				hc_add_series(
					name = "Expert Flow",
					data = purrr::transpose(chart_data),
					nodes = nodes,
					dataLabels = list(enabled = FALSE),
					tooltip = list(enabled = FALSE)
				) %>%
				hc_tooltip(enabled = FALSE) %>%
				hc_legend(enabled = FALSE) %>%
				hc_plotOptions(sankey = list(
					animation = FALSE,
					nodeWidth = 1,
					nodePadding = 10,
					curveFactor = 0.15,
					linkOpacity = 1.0,
					states = list(
						hover = list(enabled = FALSE)
					)
				)) %>%
				hc_subtitle(
					text = 'Layer Index',
					align = "center",
					verticalAlign = "bottom",
					style = list(color = '#666666', fontSize = '12px', fontWeight = 'bold'),
					y = 10 # Position near bottom
				) %>%
				hc_credits(enabled = FALSE) %>%
				hc_exporting(enabled = FALSE) %>%
				# Y-axis
				hc_chart(
					events = list(
						load = JS(sprintf("function() {
			          var chart = this;
			          // Add y-axis label
			          chart.renderer.text('%s', 20, chart.chartHeight/2)
			            .attr({rotation: 270, align: 'center'})
			            .css({color: '#666666', fontSize: '12px', fontWeight: 'bold'})
			            .add();
			        }", 'Expert ID'))
					)
				)
			
			return(hc)
		
			})
		
		return(hc_charts)
		})
	
	
	sankey_grid = hw_grid(
		all_charts %>% unlist(recursive = F),
		ncol = 3,
		rowheight = 600
		)
	
	htmltools::save_html(sankey_grid, 'routes_turbo.html', libdir = 'routes-lib')
	
	sankey_grid <<-sankey_grid
})

## Individual Paths Chart --------------------------------------------------
cluster_rate = 4

topk_plot =
	raw_routing %>%
	filter(., test_token == ':' & topk_ix %in% 1:4) %>%
	inner_join(
		.,
		sample_n(distinct(., meaning_label, batch_ix, sequence_ix, token_ix), 50),
		join_by(meaning_label, batch_ix, sequence_ix, token_ix)
	) %>%
	group_by(test_token, meaning_label, batch_ix, sequence_ix, token_ix) %>%
	mutate(., sample_ix = cur_group_id()) %>%
	ungroup() %>%
	mutate(
		.,
		expert = floor(expert / cluster_rate),
		layer_jitter = layer_ix + rnorm(nrow(.), 0, .1),
		expert_jitter = expert + rnorm(nrow(.), 0, .2)
		) %>%
	ggplot() +
	geom_line(
		aes(x = layer_jitter, y = expert_jitter, color = meaning_label, group = sample_ix), 
		alpha = .5, 
		linewidth = .5
		) +
	geom_point(
		aes(x = layer_jitter, y = expert_jitter, color = meaning_label, group = sample_ix),
		size = .5
		) +
	facet_wrap(vars(topk_ix), ncol = 1) +
	scale_x_continuous(breaks = seq(0, 26, by = 1)) +
	scale_y_continuous(breaks = seq(0, floor(63/cluster_rate), by = 1)) +
	theme_minimal() + 
	theme(
		panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
		panel.grid.minor = element_blank(),
		legend.position = 'none'
	) +
	labs(x = 'Layer index', y = 'Expert index')

feature_plots = map(group_split(raw_routing, test_token), function(token_df) {
	token_df %>%
		filter(., topk_ix %in% 1:1) %>%
		mutate(., topk_ix_str = paste0('topk = ', topk_ix)) %>%
		inner_join(
			.,
			sample_n(distinct(., meaning_label, batch_ix, sequence_ix, token_ix), 120),
			join_by(meaning_label, batch_ix, sequence_ix, token_ix)
		) %>%
		group_by(test_token, meaning_label, batch_ix, sequence_ix, token_ix) %>%
		mutate(., sample_ix = cur_group_id()) %>%
		ungroup() %>%
		mutate(
			.,
			expert = floor(expert / cluster_rate),
			layer_jitter = layer_ix + rnorm(nrow(.), 0, .1),
			expert_jitter = expert + rnorm(nrow(.), 0, .25)
		) %>%
		ggplot() +
		geom_line(
			aes(x = layer_jitter, y = expert_jitter, color = meaning_label, group = sample_ix), 
			alpha = .3, 
			linewidth = .6
		) +
		geom_point(
			aes(x = layer_jitter, y = expert_jitter, color = meaning_label, group = sample_ix),
			size = .5,
			alpha = .4
		) +
		facet_grid(cols = vars(meaning_label), rows = vars(topk_ix_str)) +
		scale_x_continuous(breaks = seq(0, 26, by = 4)) +
		scale_y_continuous(breaks = NULL) +
		ggthemes::theme_hc() + 
		theme(
			panel.grid.major = element_line(color = "gray90", linewidth = 0.5),
			panel.grid.minor = element_blank(),
			legend.position = 'none',
			axis.title = element_text(size = 9),  # Smaller axis titles
			axis.text = element_text(size = 9),   # Smaller axis labels
			strip.text = element_text(size = 10) 
			) +
		labs(title = paste0(token_df$test_token[[1]], ' routing'), x = 'Layer index', y = 'Expert index')
})

feature_plot