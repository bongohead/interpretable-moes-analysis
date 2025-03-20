library(tidyverse)
library(highcharter)

all_answers = read_csv('analyses/expert-specialization/qwen_all_answers.csv')
all_topks = read_csv('analyses/expert-specialization/qwen_all_topks.csv')

model_prefix = 'qwen_'
cluster_rate = 1
topk_to_keep = 0:3
remap_experts = F
scale_factor = 200 # Higher = smaller bands

n_layers = length(unique(all_topks$layer_ix))
n_experts = 60/cluster_rate
pad = \(x) str_pad(x, 2, 'left', '0')

# Get domain x question_ix x layer_ix level routes for the final-token ('is') for each question, topk=1
# Should validate that all tokens/token ids are identical
domain_x_q_x_layer_0 =
	all_topks %>%
	group_by(domain, question_ix) %>%
	slice_max(token_ix, with_ties = T) %>%
	ungroup() %>%
	pivot_longer(
		starts_with('expert'),
		names_to = 'topk_ix',
		values_to = 'expert_id',		
		names_transform = \(x) as.integer(str_replace_all(x, 'expert_', '')) - 1
	) %>%
	filter(topk_ix %in% topk_to_keep) %>%
	mutate(., expert_id = floor(expert_id / cluster_rate)) %>%
	select(., -index, -token_ix, -token_id, -token) 

if (remap_experts == T) {
	# Rearrange experts layer-wise by commonality
	expert_mapping =
		domain_x_q_x_layer_0 %>%
		count(layer_ix, expert_id, topk_ix) %>%
		mutate(., weighted_n = n * .1^topk_ix) %>%
		group_by(., layer_ix, expert_id) %>%
		summarize(., n = sum(weighted_n), .groups = 'drop') %>%
		group_by(., layer_ix) %>%
		mutate(new_expert_id = rank(-n, ties.method = 'first') - 1) %>% 
		ungroup()
	
	# With new expert mappings
	domain_x_q_x_layer =
		domain_x_q_x_layer_0 %>%
		left_join(expert_mapping, by = c('layer_ix', 'expert_id')) %>%
		transmute(domain, question_ix, topk_ix, layer_ix, expert_id = new_expert_id) %>%
		inner_join(all_answers, by = c('domain', 'question_ix'))
} else {
	
	domain_x_q_x_layer =
		domain_x_q_x_layer_0 %>%
		transmute(domain, question_ix, topk_ix, layer_ix, expert_id) %>%
		inner_join(all_answers, by = c('domain', 'question_ix'))
}

domain_x_q_x_layer1 = domain_x_q_x_layer %>% filter(., topk_ix == 0) %>% select(., -topk_ix)


# Table -------------------------------------------------------------------
# Last layer distribution between experts by model output
domain_x_q_x_layer1 %>%
	filter(., layer_ix == max(layer_ix) - 3) %>%
	filter(., !is.na(model_choice)) %>%
	count(., model_choice, expert_id) %>%
	pivot_wider(., names_from = expert_id, values_from = n, values_fill = 0, names_prefix = 'expert_') %>%
	View()

### TBD TOMORROW: cluster expert specialization,s ee if it correlates to either domain or question id
domain_x_q_x_layer1 %>%
	filter(., layer_ix >= max(layer_ix) & !is.na(model_choice)) %>%
	group_by(., question_ix, model_choice, domain) %>%
	arrange(layer_ix) %>%
	summarize(., expert_ids = paste0(expert_id, collapse = ','), .groups = 'drop')  %>%
	count(., expert_ids, model_choice) %>%
	pivot_wider(., names_from = expert_ids, values_from = n, values_fill = 0, names_prefix = 'expert_') %>%
	View()

domain_x_q_x_layer1 %>%
	filter(., layer_ix %in% 22:23 & !is.na(model_choice)) %>%
	pivot_wider(names_from = layer_ix, values_from = expert_id, names_prefix = 'layer_') %>%
	ggplot() + 
	geom_jitter(aes(x = layer_22, y = layer_23, color = domain), width = 2, height = 2)
# xgboost on model outputs to see if can predict

# Domain-Level Routes Routes -----------------------------------------------------
# At each domain x question x source layer x dest layer, which have the any given (source expert, dest expert)
from_layer_x_to_layer =
	inner_join(
		domain_x_q_x_layer1 %>%
			transmute(., domain, question_ix, topk_ix, from_layer = layer_ix, from_expert = expert_id),
		domain_x_q_x_layer1 %>%
			transmute(., domain, question_ix, topk_ix, to_layer = layer_ix, to_expert = expert_id, from_layer = layer_ix - 1),
		join_by(domain, question_ix, topk_ix, from_layer)
	) %>%
	group_by(domain, from_layer, to_layer, from_expert, to_expert) %>%
	summarize(., count = n(), .groups = 'drop') %>%
	group_by(domain, from_layer, to_layer) %>%
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
	cross_join(distinct(from_layer_x_to_layer, domain), .) %>%
	left_join(
		from_layer_x_to_layer,
		join_by(domain, from_layer, to_layer, from_expert, to_expert)
	) %>%
	mutate(
		from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
		to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
		count = replace_na(count, 0),
		prop = replace_na(prop, 0)
	)

hc_charts =	imap(group_split(from_layer_x_to_layer_all, domain), function(samples_for_domain, i) {
	
	sample_count = samples_for_domain %>% filter(str_detect(from, 'L00')) %>% .$count %>% sum()
	domain_label = samples_for_domain$domain[[1]]
	
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
			color = viridisLite::viridis(4)[i]
		) %>%
		purrr::transpose()
	
	chart_data =
		samples_for_domain %>%
		transmute(
			.,
			from, 
			to,
			color = ifelse(prop == 0, 'transparent', NA),
			weight = ifelse(prop == 0, .001, prop/scale_factor),
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
				'Routing for token <span style="color:deeppink; font-weight:bold; font-style:italic">is</span> ',
				'for domain <span style="color: deeppink">{domain_label}</span>',
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

grid_charts = hw_grid(hc_charts, ncol = 4, rowheight = 600)
htmltools::save_html(grid_charts, paste0(model_prefix, 'routes_by_domain.html'), libdir = 'routes-lib')

# Domain x Topk Routes Routes -----------------------------------------------------
# At each domain x question x source layer x dest layer, which have the any given (source expert, dest expert)
from_layer_x_to_layer =
	inner_join(
		domain_x_q_x_layer %>%
			transmute(., domain, question_ix, topk_ix, from_layer = layer_ix, from_expert = expert_id),
		domain_x_q_x_layer %>%
			transmute(., domain, question_ix, topk_ix, to_layer = layer_ix, to_expert = expert_id, from_layer = layer_ix - 1),
		join_by(domain, question_ix, topk_ix, from_layer)
	) %>%
	group_by(domain, topk_ix, from_layer, to_layer, from_expert, to_expert) %>%
	summarize(., count = n(), .groups = 'drop') %>%
	group_by(domain, topk_ix, from_layer, to_layer) %>%
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
	cross_join(distinct(from_layer_x_to_layer, domain, topk_ix), .) %>%
	left_join(
		from_layer_x_to_layer,
		join_by(domain, topk_ix, from_layer, to_layer, from_expert, to_expert)
	) %>%
	mutate(
		from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
		to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
		count = replace_na(count, 0),
		prop = replace_na(prop, 0)
	)

all_charts = map(group_split(from_layer_x_to_layer_all, topk_ix), function(samples_for_topk) {
	topk = samples_for_topk$topk_ix[[1]]
	print(topk)
	
	hc_charts =	imap(group_split(samples_for_topk, domain), function(samples_for_domain, i) {
		
		sample_count = samples_for_domain %>% filter(str_detect(from, 'L00')) %>% .$count %>% sum()
		domain_label = samples_for_domain$domain[[1]]
		
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
				color = viridisLite::viridis(4)[i]
			) %>%
			purrr::transpose()
		
		chart_data =
			samples_for_domain %>%
			transmute(
				.,
				from, 
				to,
				color = ifelse(prop == 0, 'transparent', NA),
				weight = ifelse(prop == 0, .001, prop/scale_factor),
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
					'Topk = {topk} routing for token <span style="color:deeppink; font-weight:bold; font-style:italic">is</span> ',
					'for domain <span style="color: deeppink">{domain_label}</span>',
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

grid_charts = hw_grid(all_charts %>% unlist(recursive = F), ncol = 4, rowheight = 600)
htmltools::save_html(grid_charts, paste0(model_prefix, 'routes_by_domain_topk.html'), libdir = 'routes-lib')

# Solution-Level Routes -----------------------------------------------------
# At each domain x question x source layer x dest layer, which have the any given (source expert, dest expert)
from_layer_x_to_layer =
	inner_join(
		domain_x_q_x_layer1 %>%
			transmute(., domain, question_ix, topk_ix, model_choice, from_layer = layer_ix, from_expert = expert_id),
		domain_x_q_x_layer1 %>%
			transmute(., domain, question_ix, topk_ix, model_choice, to_layer = layer_ix, to_expert = expert_id, from_layer = layer_ix - 1),
		join_by(domain, question_ix, topk_ix, model_choice, from_layer)
	) %>%
	group_by(model_choice, from_layer, to_layer, from_expert, to_expert) %>%
	summarize(., count = n(), .groups = 'drop') %>%
	group_by(model_choice, from_layer, to_layer) %>%
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
	cross_join(distinct(from_layer_x_to_layer, model_choice), .) %>%
	left_join(
		from_layer_x_to_layer,
		join_by(model_choice, from_layer, to_layer, from_expert, to_expert)
	) %>%
	mutate(
		from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
		to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
		count = replace_na(count, 0),
		prop = replace_na(prop, 0)
	) %>%
	na.omit()

hc_charts =	imap(group_split(from_layer_x_to_layer_all, model_choice), function(sample_for_choice, i) {
	
	sample_count = sample_for_choice %>% filter(str_detect(from, 'L00')) %>% .$count %>% sum()
	domain_label = sample_for_choice$model_choice[[1]]
	
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
			color = viridisLite::viridis(4)[i]
		) %>%
		purrr::transpose()
	
	chart_data =
		sample_for_choice %>%
		transmute(
			.,
			from, 
			to,
			color = ifelse(prop == 0, 'transparent', NA),
			weight = ifelse(prop == 0, .001, prop/scale_factor),
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
				'Routing for token <span style="color:deeppink; font-weight:bold; font-style:italic">is</span> ',
				'for domain <span style="color: deeppink">{domain_label}</span>',
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

grid_charts = hw_grid(hc_charts, ncol = 4, rowheight = 600)
htmltools::save_html(grid_charts, paste0(model_prefix, 'routes_by_solution.html'), libdir = 'routes-lib')

# Solution x Topk Routes Routes -----------------------------------------------------
# At each domain x question x source layer x dest layer, which have the any given (source expert, dest expert)
from_layer_x_to_layer =
	inner_join(
		domain_x_q_x_layer %>%
			transmute(., domain, question_ix, topk_ix, model_choice, from_layer = layer_ix, from_expert = expert_id),
		domain_x_q_x_layer %>%
			transmute(., domain, question_ix, topk_ix, model_choice, to_layer = layer_ix, to_expert = expert_id, from_layer = layer_ix - 1),
		join_by(domain, question_ix, model_choice, topk_ix, from_layer)
	) %>%
	group_by(model_choice, topk_ix, from_layer, to_layer, from_expert, to_expert) %>%
	summarize(., count = n(), .groups = 'drop') %>%
	group_by(model_choice, topk_ix, from_layer, to_layer) %>%
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
	cross_join(distinct(from_layer_x_to_layer, model_choice, topk_ix), .) %>%
	left_join(
		from_layer_x_to_layer,
		join_by(model_choice, topk_ix, from_layer, to_layer, from_expert, to_expert)
	) %>%
	mutate(
		from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
		to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
		count = replace_na(count, 0),
		prop = replace_na(prop, 0)
	) %>%
	na.omit()

all_charts = map(group_split(from_layer_x_to_layer_all, topk_ix), function(samples_for_topk) {
	topk = samples_for_topk$topk_ix[[1]]
	print(topk)
	
	hc_charts =	imap(group_split(samples_for_topk, model_choice), function(samples_for_domain, i) {
		
		sample_count = samples_for_domain %>% filter(str_detect(from, 'L00')) %>% .$count %>% sum()
		domain_label = samples_for_domain$model_choice[[1]]
		
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
				color = viridisLite::viridis(4)[i]
			) %>%
			purrr::transpose()
		
		chart_data =
			samples_for_domain %>%
			transmute(
				.,
				from, 
				to,
				color = ifelse(prop == 0, 'transparent', NA),
				weight = ifelse(prop == 0, .001, prop/scale_factor),
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
					'Topk = {topk} routing for token <span style="color:deeppink; font-weight:bold; font-style:italic">is</span> ',
					'for answer <span style="color: deeppink">{domain_label}</span>',
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

grid_charts = hw_grid(all_charts %>% unlist(recursive = F), ncol = 4, rowheight = 500)
htmltools::save_html(grid_charts, paste0(model_prefix, 'routes_by_solution_topk.html'), libdir = 'routes-lib')

# Correct x Topk Routes Routes -----------------------------------------------------
# At each domain x question x source layer x dest layer, which have the any given (source expert, dest expert)
from_layer_x_to_layer =
	inner_join(
		domain_x_q_x_layer %>%
			transmute(., domain, question_ix, topk_ix, is_correct, from_layer = layer_ix, from_expert = expert_id),
		domain_x_q_x_layer %>%
			transmute(., domain, question_ix, topk_ix, is_correct, to_layer = layer_ix, to_expert = expert_id, from_layer = layer_ix - 1),
		join_by(domain, question_ix, is_correct, topk_ix, from_layer)
	) %>%
	group_by(is_correct, topk_ix, from_layer, to_layer, from_expert, to_expert) %>%
	summarize(., count = n(), .groups = 'drop') %>%
	group_by(is_correct, topk_ix, from_layer, to_layer) %>%
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
	cross_join(distinct(from_layer_x_to_layer, is_correct, topk_ix), .) %>%
	left_join(
		from_layer_x_to_layer,
		join_by(is_correct, topk_ix, from_layer, to_layer, from_expert, to_expert)
	) %>%
	mutate(
		from = paste0("L", pad(from_layer), "_E", pad(from_expert)),
		to = paste0("L", pad(to_layer), "_E", pad(to_expert)),
		count = replace_na(count, 0),
		prop = replace_na(prop, 0)
	)

all_charts = map(group_split(from_layer_x_to_layer_all, topk_ix), function(samples_for_topk) {
	topk = samples_for_topk$topk_ix[[1]]
	print(topk)
	
	hc_charts =	imap(group_split(samples_for_topk, is_correct), function(samples_for_domain, i) {
		
		sample_count = samples_for_domain %>% filter(str_detect(from, 'L00')) %>% .$count %>% sum()
		domain_label = samples_for_domain$is_correct[[1]]
		
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
				color = viridisLite::viridis(2)[i]
			) %>%
			purrr::transpose()
		
		chart_data =
			samples_for_domain %>%
			transmute(
				.,
				from, 
				to,
				color = ifelse(prop == 0, 'transparent', NA),
				weight = ifelse(prop == 0, .001, prop/scale_factor),
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
					'Topk = {topk} routing for token <span style="color:deeppink; font-weight:bold; font-style:italic">is</span> ',
					'for answer <span style="color: deeppink">{domain_label}</span>',
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

grid_charts = hw_grid(all_charts %>% unlist(recursive = F), ncol = 2, rowheight = 500)
htmltools::save_html(grid_charts, paste0(model_prefix, 'routes_by_correctness_topk.html'), libdir = 'routes-lib')
