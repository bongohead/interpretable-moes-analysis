library(tidyverse)
library(highcharter)

raw_routing = read_csv('analyses/export_0.csv')
cluster_rate = 4

n_layers = length(unique(raw_routing$layer))
n_experts = 64/cluster_rate
pad = \(x) str_pad(x, 2, 'left', '0')

sample_by_layer_routing =
	raw_routing %>%
	pivot_longer(
		starts_with('expert'),
		names_to = 'topk_ix',
		values_to = 'expert_id',		
		names_transform = \(x) as.integer(str_replace_all(x, 'expert_', '')) - 1
	) %>%
	filter(token_id == 27 & topk_ix %in% 0:0) %>%
	group_by(prompt_ix, batch_ix, token_ix) %>%
	mutate(., sample_ix = cur_group_id()) %>%
	ungroup() %>%
	select(., sample_ix, layer, expert_id) %>%
	mutate(., expert_id = floor(expert_id / cluster_rate))

# Get proportion of all layer which have the any given (source expert, dest expert)
routing_map_counts = 
	inner_join(
		sample_by_layer_routing %>%
			transmute(., sample_ix, from_layer = layer, from_expert = expert_id),
		sample_by_layer_routing %>%
			transmute(., sample_ix, to_layer = layer, to_expert = expert_id, from_layer = layer - 1),
		join_by(sample_ix, from_layer)
	) %>%
	group_by(from_layer, to_layer, from_expert, to_expert) %>%
	summarize(., count = n(), .groups = 'drop') %>%
	group_by(from_layer, to_layer) %>%
	mutate(., prop = count/sum(count)) %>%
	ungroup()

# Join in empty data
routing_map =
	inner_join(
		crossing(from_layer = 0:(n_layers - 2), from_expert = 0:(n_experts - 1)),
		crossing(to_layer = 1:(n_layers - 1), to_expert = 0:(n_experts - 1)) %>%
			mutate(., from_layer = to_layer - 1),
		join_by(from_layer),
		relationship = 'many-to-many'
	) %>%
	left_join(routing_map_counts, join_by(from_layer, to_layer, from_expert, to_expert)) %>%
	mutate(
		from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
		to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
		count = replace_na(count, 0),
		prop = replace_na(prop, 0)
	)
	
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
	routing_map %>%
	transmute(
		.,
		from, 
		to,
		color = ifelse(prop == 0, 'transparent', NA),
		weight = ifelse(prop == 0, .001, prop/50),
		)
	
highchart() %>%
	hc_chart(
		type = "sankey",
		marginLeft = 40,  # Add left margin for y-axis label
		marginBottom = 40 # Add bottom margin for x-axis label
		) %>%
	hc_title(text = "Expert Routing for Token `:`, Context 2") %>%
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
		y = 20 # Position near bottom
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
            .attr({
              rotation: 270,
              align: 'center'
            })
            .css({
              color: '#666666',
              fontSize: '12px',
              fontWeight: 'bold'
            })
            .add();
        }", 'Expert ID'))
		)
	)

