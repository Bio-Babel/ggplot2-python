# API Reference

## Core Functions

| Function | Description |
|---|---|
| `ggplot(data, mapping)` | Create a new ggplot object |
| `aes(x, y, ...)` | Create aesthetic mappings |
| `after_stat(x)` | Reference computed stat variables in aesthetics |
| `after_scale(x)` | Reference scaled aesthetic values |
| `stage(start, after_stat, after_scale)` | Control aesthetic evaluation stages |
| `vars(...)` | Quote variables for faceting |
| `ggsave(filename, plot)` | Save a plot to file |
| `qplot(x, y, ...)` | Quick plot (convenience wrapper) |
| `last_plot()` | Retrieve the last plot created |
| `layer(...)` | Create a layer directly |

## Geoms

| Function | Description |
|---|---|
| `geom_point()` | Scatter plot |
| `geom_path()` | Connect observations in data order |
| `geom_line()` | Connect observations in x order |
| `geom_step()` | Step plot |
| `geom_bar()` | Bar chart (stat=count) |
| `geom_col()` | Bar chart (stat=identity) |
| `geom_rect()` | Rectangles |
| `geom_tile()` | Rectangles via center + dimensions |
| `geom_raster()` | High-performance rectangular tiling |
| `geom_text()` | Text labels |
| `geom_label()` | Text labels with background |
| `geom_boxplot()` | Box-and-whisker plot |
| `geom_violin()` | Violin plot |
| `geom_dotplot()` | Dot plot |
| `geom_ribbon()` | Ribbons (filled area between ymin and ymax) |
| `geom_area()` | Area plot (ribbon anchored to y=0) |
| `geom_smooth()` | Smoothed conditional means |
| `geom_polygon()` | Polygons |
| `geom_errorbar()` | Vertical error bars |
| `geom_errorbarh()` | Horizontal error bars |
| `geom_crossbar()` | Crossbar (box with middle line) |
| `geom_linerange()` | Vertical line range |
| `geom_pointrange()` | Point with vertical line range |
| `geom_segment()` | Line segments |
| `geom_curve()` | Curved line segments |
| `geom_spoke()` | Line segments defined by angle and radius |
| `geom_density()` | Kernel density estimate |
| `geom_density2d()` | 2D density contour lines |
| `geom_density2d_filled()` | Filled 2D density contours |
| `geom_contour()` | Contour lines of 3D surface |
| `geom_contour_filled()` | Filled contours of 3D surface |
| `geom_hex()` | Hexagonal binning |
| `geom_bin2d()` | 2D rectangular binning |
| `geom_abline()` | Reference lines (slope + intercept) |
| `geom_hline()` | Horizontal reference line |
| `geom_vline()` | Vertical reference line |
| `geom_rug()` | Marginal rug plot |
| `geom_blank()` | Empty geom (for expanding limits) |
| `geom_function()` | Draw a function as a line |
| `geom_histogram()` | Histogram |
| `geom_freqpoly()` | Frequency polygon |
| `geom_count()` | Count overlapping points |
| `geom_jitter()` | Jittered scatter plot |
| `geom_map()` | Polygons from a map data frame |
| `geom_quantile()` | Quantile regression lines |
| `geom_sf()` | Render simple features |
| `geom_sf_label()` | Label simple features |
| `geom_sf_text()` | Text labels for simple features |
| `geom_qq()` | Quantile-quantile points |
| `geom_qq_line()` | Quantile-quantile reference line |

## Stats

| Function | Description |
|---|---|
| `stat_identity()` | Identity transformation (no computation) |
| `stat_bin()` | Binning for histograms |
| `stat_count()` | Count unique values |
| `stat_density()` | 1D kernel density estimate |
| `stat_smooth()` | Smoothed conditional means |
| `stat_boxplot()` | Box plot statistics |
| `stat_summary()` | Summarise y at each x |
| `stat_summary_bin()` | Summarise y in bins of x |
| `stat_summary2d()` / `stat_summary_2d()` | 2D summary |
| `stat_summary_hex()` | Hexagonal 2D summary |
| `stat_function()` | Compute function values |
| `stat_ecdf()` | Empirical cumulative distribution function |
| `stat_qq()` | Quantile-quantile calculation |
| `stat_qq_line()` | Quantile-quantile reference line calculation |
| `stat_bin2d()` / `stat_bin_2d()` | 2D rectangular binning |
| `stat_bin_hex()` / `stat_binhex()` | Hexagonal binning |
| `stat_contour()` | Contour calculation |
| `stat_contour_filled()` | Filled contour calculation |
| `stat_density2d()` / `stat_density_2d()` | 2D kernel density |
| `stat_density2d_filled()` / `stat_density_2d_filled()` | Filled 2D kernel density |
| `stat_ellipse()` | Confidence ellipse |
| `stat_unique()` | Remove duplicates |
| `stat_sum()` | Count overlapping points |
| `stat_ydensity()` | Violin density estimate |
| `stat_align()` | Align observations across groups |
| `stat_connect()` | Connect observations |
| `stat_manual()` | Manual stat (user-supplied computation) |
| `stat_quantile()` | Quantile regression |
| `stat_spoke()` | Convert angle/radius to segment endpoints |

### Summary helpers

| Function | Description |
|---|---|
| `mean_se()` | Mean and standard error |
| `mean_cl_boot()` | Mean with bootstrapped confidence limits |
| `mean_cl_normal()` | Mean with normal confidence limits |
| `mean_sdl()` | Mean +/- constant * standard deviation |
| `median_hilow()` | Median with high and low quantiles |

## Scales

### Position scales

| Function | Description |
|---|---|
| `scale_x_continuous()` / `scale_y_continuous()` | Continuous position scale |
| `scale_x_discrete()` / `scale_y_discrete()` | Discrete position scale |
| `scale_x_log10()` / `scale_y_log10()` | Log10-transformed position |
| `scale_x_sqrt()` / `scale_y_sqrt()` | Square-root-transformed position |
| `scale_x_reverse()` / `scale_y_reverse()` | Reversed position |
| `scale_x_binned()` / `scale_y_binned()` | Binned position |
| `scale_x_date()` / `scale_y_date()` | Date position |
| `scale_x_datetime()` / `scale_y_datetime()` | Datetime position |
| `scale_x_time()` / `scale_y_time()` | Time position |

### Colour and fill scales

| Function | Description |
|---|---|
| `scale_colour_continuous()` / `scale_fill_continuous()` | Continuous colour mapping |
| `scale_colour_discrete()` / `scale_fill_discrete()` | Discrete colour mapping |
| `scale_colour_gradient()` / `scale_fill_gradient()` | Sequential two-colour gradient |
| `scale_colour_gradient2()` / `scale_fill_gradient2()` | Diverging three-colour gradient |
| `scale_colour_gradientn()` / `scale_fill_gradientn()` | N-colour gradient |
| `scale_colour_hue()` / `scale_fill_hue()` | Evenly-spaced hue colours |
| `scale_colour_brewer()` / `scale_fill_brewer()` | ColorBrewer palettes |
| `scale_colour_distiller()` / `scale_fill_distiller()` | Continuous ColorBrewer palettes |
| `scale_colour_fermenter()` / `scale_fill_fermenter()` | Binned ColorBrewer palettes |
| `scale_colour_viridis_c()` / `scale_fill_viridis_c()` | Continuous viridis palettes |
| `scale_colour_viridis_d()` / `scale_fill_viridis_d()` | Discrete viridis palettes |
| `scale_colour_viridis_b()` / `scale_fill_viridis_b()` | Binned viridis palettes |
| `scale_colour_grey()` / `scale_fill_grey()` | Grey palette |
| `scale_colour_identity()` / `scale_fill_identity()` | Use values as-is |
| `scale_colour_manual()` / `scale_fill_manual()` | Manual colour values |
| `scale_colour_binned()` / `scale_fill_binned()` | Binned colour mapping |
| `scale_colour_steps()` / `scale_fill_steps()` | Binned sequential gradient |
| `scale_colour_steps2()` / `scale_fill_steps2()` | Binned diverging gradient |
| `scale_colour_stepsn()` / `scale_fill_stepsn()` | Binned N-colour gradient |
| `scale_colour_date()` / `scale_fill_date()` | Date colour mapping |
| `scale_colour_datetime()` / `scale_fill_datetime()` | Datetime colour mapping |
| `scale_colour_ordinal()` / `scale_fill_ordinal()` | Ordinal colour mapping |

All `scale_colour_*` functions have `scale_color_*` aliases.

### Alpha scales

| Function | Description |
|---|---|
| `scale_alpha()` | Continuous alpha (transparency) |
| `scale_alpha_continuous()` | Continuous alpha |
| `scale_alpha_discrete()` | Discrete alpha |
| `scale_alpha_binned()` | Binned alpha |
| `scale_alpha_identity()` | Use alpha values as-is |
| `scale_alpha_manual()` | Manual alpha values |
| `scale_alpha_ordinal()` | Ordinal alpha |
| `scale_alpha_date()` | Date-based alpha |
| `scale_alpha_datetime()` | Datetime-based alpha |

### Size scales

| Function | Description |
|---|---|
| `scale_size()` | Map variable to point size |
| `scale_size_continuous()` | Continuous size |
| `scale_size_discrete()` | Discrete size |
| `scale_size_binned()` | Binned size |
| `scale_size_area()` | Size by area (0 maps to 0) |
| `scale_size_binned_area()` | Binned area-proportional size |
| `scale_size_identity()` | Use size values as-is |
| `scale_size_manual()` | Manual size values |
| `scale_size_ordinal()` | Ordinal size |
| `scale_size_date()` | Date-based size |
| `scale_size_datetime()` | Datetime-based size |
| `scale_radius()` | Map variable to radius |

### Shape scales

| Function | Description |
|---|---|
| `scale_shape()` | Map variable to point shape |
| `scale_shape_discrete()` | Discrete shapes |
| `scale_shape_binned()` | Binned shapes |
| `scale_shape_identity()` | Use shape values as-is |
| `scale_shape_manual()` | Manual shape values |
| `scale_shape_ordinal()` | Ordinal shapes |

### Linetype scales

| Function | Description |
|---|---|
| `scale_linetype()` | Map variable to line type |
| `scale_linetype_discrete()` | Discrete line types |
| `scale_linetype_binned()` | Binned line types |
| `scale_linetype_identity()` | Use linetype values as-is |
| `scale_linetype_manual()` | Manual line type values |

### Linewidth scales

| Function | Description |
|---|---|
| `scale_linewidth()` | Map variable to line width |
| `scale_linewidth_continuous()` | Continuous linewidth |
| `scale_linewidth_discrete()` | Discrete linewidth |
| `scale_linewidth_binned()` | Binned linewidth |
| `scale_linewidth_identity()` | Use linewidth values as-is |
| `scale_linewidth_manual()` | Manual linewidth values |
| `scale_linewidth_ordinal()` | Ordinal linewidth |
| `scale_linewidth_date()` | Date-based linewidth |
| `scale_linewidth_datetime()` | Datetime-based linewidth |

### Scale infrastructure

| Function | Description |
|---|---|
| `continuous_scale()` | Create a continuous scale |
| `discrete_scale()` | Create a discrete scale |
| `binned_scale()` | Create a binned scale |
| `sec_axis()` | Specify a secondary axis |
| `dup_axis()` | Duplicate primary axis on secondary |
| `expansion()` / `expand_scale()` | Control scale expansion |
| `find_scale()` | Find appropriate scale for an aesthetic |
| `scale_type()` | Determine scale type for data |

## Coordinates

| Function | Description |
|---|---|
| `coord_cartesian()` | Cartesian coordinates (default) |
| `coord_fixed()` / `coord_equal()` | Fixed aspect ratio |
| `coord_flip()` | Flip x and y axes |
| `coord_polar()` | Polar coordinates |
| `coord_radial()` | Radial coordinates |
| `coord_trans()` / `coord_transform()` | Transformed Cartesian coordinates |
| `coord_munch()` | Munch coordinates for curved geometries |

## Facets

| Function | Description |
|---|---|
| `facet_grid()` | Lay out panels in a grid (rows ~ cols) |
| `facet_wrap()` | Wrap 1D ribbon of panels |
| `facet_null()` | No faceting (default) |

## Position Adjustments

| Function | Description |
|---|---|
| `position_identity()` | No adjustment |
| `position_dodge()` | Dodge overlapping objects side-to-side |
| `position_dodge2()` | Dodge preserving total width |
| `position_jitter()` | Randomly jitter points |
| `position_jitterdodge()` | Dodge and jitter |
| `position_nudge()` | Nudge points by fixed offset |
| `position_stack()` | Stack overlapping objects |
| `position_fill()` | Stack and normalise to fill |

## Themes

### Complete themes

| Function | Description |
|---|---|
| `theme_grey()` / `theme_gray()` | Default grey theme |
| `theme_bw()` | Black-and-white theme |
| `theme_linedraw()` | Line drawing theme |
| `theme_light()` | Light theme |
| `theme_dark()` | Dark theme |
| `theme_minimal()` | Minimal theme |
| `theme_classic()` | Classic theme (axes only) |
| `theme_void()` | Empty theme |
| `theme_test()` | Theme for visual testing |

### Theme elements

| Function | Description |
|---|---|
| `element_blank()` | Remove element |
| `element_line()` | Style lines |
| `element_rect()` | Style rectangles |
| `element_text()` | Style text |
| `element_point()` | Style points |
| `element_polygon()` | Style polygons |
| `element_geom()` | Default geom styling |
| `element_grob()` | Custom grob element |

### Theme utilities

| Function | Description |
|---|---|
| `theme(...)` | Modify theme elements |
| `theme_get()` / `get_theme()` | Get current default theme |
| `theme_set()` / `set_theme()` | Set default theme |
| `theme_update()` / `update_theme()` | Update default theme |
| `theme_replace()` / `replace_theme()` | Replace default theme elements |
| `reset_theme_settings()` | Reset theme to default |
| `margin()` | Define margins (top, right, bottom, left) |
| `rel()` | Relative sizing |
| `calc_element()` | Calculate element from theme hierarchy |
| `merge_element()` | Merge two theme elements |

## Guides

| Function | Description |
|---|---|
| `guide_axis()` | Axis guide |
| `guide_axis_logticks()` | Axis with log-scale tick marks |
| `guide_axis_stack()` | Stacked axis guides |
| `guide_axis_theta()` | Theta axis for polar coordinates |
| `guide_legend()` | Legend guide |
| `guide_colourbar()` / `guide_colorbar()` | Continuous colour bar |
| `guide_coloursteps()` / `guide_colorsteps()` | Binned colour steps |
| `guide_bins()` | Binned legend |
| `guide_custom()` | Custom guide grob |
| `guide_none()` | Suppress guide |
| `guides(...)` | Set guides for multiple aesthetics |

## Labels and Annotations

| Function | Description |
|---|---|
| `labs(...)` | Set plot labels (title, subtitle, x, y, caption, etc.) |
| `xlab()` | Set x-axis label |
| `ylab()` | Set y-axis label |
| `ggtitle()` | Set plot title and subtitle |
| `lims(...)` | Set scale limits |
| `xlim()` | Set x-axis limits |
| `ylim()` | Set y-axis limits |
| `expand_limits()` | Expand limits to include values |
| `annotate()` | Add annotation layer |
| `annotation_custom()` | Fixed-position grob annotation |
| `annotation_raster()` | Fixed-position raster annotation |
| `annotation_logticks()` | Log-scale tick mark annotations |

## Draw Keys

| Function | Description |
|---|---|
| `draw_key_point()` | Legend key for points |
| `draw_key_path()` | Legend key for paths/lines |
| `draw_key_rect()` | Legend key for rectangles |
| `draw_key_polygon()` | Legend key for polygons |
| `draw_key_blank()` | Empty legend key |
| `draw_key_boxplot()` | Legend key for box plots |
| `draw_key_crossbar()` | Legend key for crossbars |
| `draw_key_dotplot()` | Legend key for dot plots |
| `draw_key_label()` | Legend key for labels |
| `draw_key_linerange()` | Legend key for line ranges |
| `draw_key_pointrange()` | Legend key for point ranges |
| `draw_key_smooth()` | Legend key for smoothed lines |
| `draw_key_text()` | Legend key for text |
| `draw_key_abline()` | Legend key for reference lines |
| `draw_key_vline()` | Legend key for vertical lines |
| `draw_key_timeseries()` | Legend key for time series |
| `draw_key_vpath()` | Legend key for vertical paths |

## Plot Introspection

| Function | Description |
|---|---|
| `ggplot_build(plot)` | Build a plot for rendering |
| `ggplot_gtable(data)` | Convert built plot data to gtable |
| `ggplotGrob(plot)` | Convert plot to grob |
| `get_layer_data(plot, i)` | Extract computed data from a layer |
| `get_layer_grob(plot, i)` | Extract grob from a layer |
| `get_panel_scales(plot)` | Extract panel scales |
| `get_guide_data(plot)` | Extract guide data |
| `get_strip_labels(plot)` | Extract facet strip labels |
| `get_labs(plot)` | Extract all labels |
| `summarise_plot(plot)` | Summary of plot structure |
| `summarise_coord(plot)` | Summary of coordinate system |
| `summarise_layers(plot)` | Summary of layers |
| `summarise_layout(plot)` | Summary of layout |

## Datasets

| Dataset | Description |
|---|---|
| `datasets.diamonds` | Prices of 50,000+ round-cut diamonds |
| `datasets.economics` | US economic time series |
| `datasets.economics_long` | Economics data in long format |
| `datasets.faithfuld` | Old Faithful eruption data (2D density) |
| `datasets.luv_colours` | Colour data in Luv space |
| `datasets.midwest` | Midwest demographics |
| `datasets.mpg` | Fuel economy data (1999-2008) |
| `datasets.msleep` | Mammalian sleep data |
| `datasets.presidential` | US presidential terms |
| `datasets.seals` | Seal movement vectors |
| `datasets.txhousing` | Texas housing market data |
