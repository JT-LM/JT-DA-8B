class Function_Calls:
    def __init__(self):

        self.create_chart_thought = """
这是一个图表可视化工具，用于创建各种图表，可以参考调用示例中的.create方法进行使用。以下是这个工具的说明：

### 功能描述
```python
chart_tool = ChartTool()
chart_tool.create(chart_type=chart_type, data=data, global_opts=global_opts)
```
此工具通过初化一个PyechartsFunction类（已定义好，可以直接调用），并通过.create()函数生成并保存图表文件。支持以下图表类型：
bar(柱状图), line(折线图), boxplot(箱线图), scatter(散点图), 
wordcloud(词云), geo(地理图), gauge(仪表盘), liquid(水球图), 
pictorialbar(象形柱图), pie(饼图)

### 参数规范
调用create()需提供的调用参数：

1. chart_type (必填)
   - 类型: string
   - 可选值: ["bar", "line", "boxplot", "scatter", "wordcloud", "geo", "gauge", "liquid", "pictorialbar", "pie"]

2. data (必填)
   - 类型: object
   - 结构:
     {
       "categories": [str]   # X轴分类(柱状/折线图等需要)
       "series": [
         {
           "name": "系列名称", 
           "data": [...]     # 数据类型根据图表变化:
             - 柱/折/散点: [num1, num2, ...] 
             - 箱线图: [[min,Q1,median,Q3,max], ...] 
             - 饼/词云/地理: [{"name":str, "value":num}, ...]
         }
       ]
     }

3. global_opts (可选)
   - 类型: object
   - 配置示例:
     {
       "init": {"width": "900px", "height": "500px", "theme": "dark"},
       "title": {"title": "销售数据", "pos_left": "left"},
       "legend": {"is_show": true, "pos_top": "10%"},
       "tooltip": {"trigger": "axis"},
       "xaxis": {"name": "月份"},
       "yaxis": {"name": "销售额"}
     }

4. series_opts (可选)
   - 类型: object
   - 示例: {"label_opts": {"is_show": true}, "color": "#FF0000"}

5. render_path (可选)
   - 类型: string
   - 默认: "chart.html"

### 注意事项
1. 在调用时需要先pyecharts_func = PyechartsFunction()初始化此实例
2. 地理图(geo)需在global_opts中添加:
   "geo": {"maptype": "china", ...}  # maptype支持省份/国家名
3. 饼图/词云不需要categories字段
4. 主题支持: "light"(默认), "dark", "chalk", "essos", "macarons"
5. 函数返回值为生成的HTML文件绝对路径

### 调用示例
用户请求："创建2023年各季度销售额柱状图"
调用参数:
{
  "chart_type": "bar",
  "data": {
    "categories": ["Q1", "Q2", "Q3", "Q4"],
    "series": [
      {"name": "销售额", "data": [120, 150, 180, 200]}
    ]
  },
  "global_opts": {
    "title": {"title": "2023季度销售报告"},
    "xaxis": {"name": "季度"},
    "yaxis": {"name": "万元"}
  }
}
chart_tool = ChartTool()
chart_tool.create(chart_type=chart_type, data=data, global_opts=global_opts)"""
    
        self.create_chart_call = """
    import os
    import copy
    import inspect
    from typing import Any, Dict
    
    
    from pyecharts import options as opts
    from pyecharts.charts import (
        Bar, Line, Boxplot, Scatter, WordCloud, Geo, Gauge, Liquid, PictorialBar, Pie
    )
    from pyecharts.globals import ThemeType, ChartType
    
    def _deep_merge_dicts(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                # 如果源和目标都是字典，则递归合并
                destination[key] = _deep_merge_dicts(destination[key], value)
            else:
                # 否则，源值覆盖目标值
                destination[key] = value
        return destination
    
    
    class ChartTool:
        def __init__(self):
            self.DEFAULT_GLOBAL_OPTS = {
                "init": {"width": "760px", "height": "400px", "theme": "light"},
                "title": {"title": "默认标题", "pos_left": "center"},
                "legend": {"is_show": True, "pos_top": "20%"},
                "tooltip": {"trigger": "item", "trigger_on":"mousemove|click", "axis_pointer_type": "shadow"},
                "toolbox": {"is_show": True},
                "xaxis": {"name": "X轴"},
                "yaxis": {"name": "Y轴"}
            }
            self.theme_mapping = {
                "light": ThemeType.LIGHT, "dark": ThemeType.DARK, "chalk": ThemeType.CHALK,
                "essos": ThemeType.ESSOS, "macarons": ThemeType.MACARONS,
            }
            self.chart_map = {
                "bar": Bar, "line": Line, "boxplot": Boxplot, "scatter": Scatter,
                "wordcloud": WordCloud, "geo": Geo, "gauge": Gauge, "liquid": Liquid,
                "pictorialbar": PictorialBar, "pie": Pie
            }
            self.current_chart = None
            self.overlap_charts = []
        
        def _build_opts(self, opt_class: Any, config: Dict) -> Any:
            if not config:
                return None
            try:
                sig = inspect.signature(opt_class.__init__)
                valid_params = set(sig.parameters.keys())
            except (TypeError, AttributeError):
                # 如果 opt_class 不是一个合适的类，则无法过滤，按原样尝试
                return opt_class(**config)
            
            filtered_config = {k: v for k, v in config.items() if k in valid_params}
            removed_keys = set(config.keys()) - set(filtered_config.keys())
            if removed_keys:
                print(f"提示: 在为 {opt_class.__name__} 创建选项时，以下无效键被忽略: {removed_keys}")
    
            return opt_class(**filtered_config)
        
        def _initialize_chart(self, chart_type: str, init_opts_config: Dict) -> Any:
            if chart_type not in self.chart_map:
                raise ValueError(f"不支持的图表类型: {chart_type}")
                
            theme_key = init_opts_config.get("theme", "light")
            init_opts_config["theme"] = self.theme_mapping.get(theme_key, ThemeType.LIGHT)
            init_opts_config["width"] = "760px"
            init_opts_config["height"] = "400px"
            init_opts = self._build_opts(opts.InitOpts, init_opts_config)
            chart_class = self.chart_map[chart_type]
            self.current_chart = chart_class(init_opts=init_opts)
            return self.current_chart
    
        def _apply_global_options(self, chart_instance: Any, global_opts_config: Dict):
            if not chart_instance:
                raise RuntimeError("图表实例尚未初始化。")
            
            opts_map = {
                "title_opts": (opts.TitleOpts, global_opts_config.get("title")),
                "legend_opts": (opts.LegendOpts, global_opts_config.get("legend")),
                "tooltip_opts": (opts.TooltipOpts, global_opts_config.get("tooltip")),
                "toolbox_opts": (opts.ToolboxOpts, global_opts_config.get("toolbox")),
                "xaxis_opts": (opts.AxisOpts, global_opts_config.get("xaxis")),
                "yaxis_opts": (opts.AxisOpts, global_opts_config.get("yaxis")),
            }
            
            built_opts = {
                key: self._build_opts(opt_class, config)
                for key, (opt_class, config) in opts_map.items() if config
            }
    
            if isinstance(chart_instance, Geo):
                geo_opts_config = global_opts_config.get("geo")
                if geo_opts_config:
                    maptype = geo_opts_config.pop("maptype", "china")
                    itemstyle_opts_config = geo_opts_config.pop("itemstyle_opts", {})
                    itemstyle_opts = self._build_opts(opts.ItemStyleOpts, itemstyle_opts_config)
                    chart_instance.add_schema(maptype=maptype, itemstyle_opts=itemstyle_opts, **geo_opts_config)
            
            chart_instance.set_global_opts(**built_opts) 
        
        def _add_series_data(self, chart_instance: Any, chart_type: str, data: Dict, series_opts_config: Dict):
    
            if not chart_instance:
                raise RuntimeError("图表实例尚未初始化。")
    
            categories = data.get("categories")
            categories = [str(i) for i in categories]
            if categories and hasattr(chart_instance, "add_xaxis"):
                chart_instance.add_xaxis(categories)
    
            for s in data.get("series", []):
                series_name = s.get("name", "")
                series_values = s.get("data", [])
                
                # print("series_name", series_name, type(series_name))
                # print("series_values", series_values, type(series_values))
    
                final_series_opts = copy.deepcopy(series_opts_config)
                specific_opts = {k: v for k, v in s.items() if k not in ["name", "data"]}
                _deep_merge_dicts(final_series_opts, specific_opts)
                add_method = None
                if chart_type in ["bar", "line", "scatter", "pictorialbar", "boxplot"]:
                    add_method = chart_instance.add_yaxis
                else:
                    add_method = chart_instance.add
    
                sig = inspect.signature(add_method)
                valid_params = set(sig.parameters.keys())
                filtered_series_opts = {k: v for k, v in final_series_opts.items() if k in valid_params}
    
                removed_keys = set(final_series_opts.keys()) - set(filtered_series_opts.keys())
                if removed_keys:
                    print(f"提示: 在为系列 '{series_name}' 添加数据时，以下无效系列配置键被忽略: {removed_keys}")
    
                if chart_type in ["bar", "line", "scatter", "pictorialbar"]:
                    chart_instance.add_yaxis(series_name, series_values, **filtered_series_opts)
                elif chart_type == "boxplot":
                    prepared_data = chart_instance.prepare_data(series_values)
                    chart_instance.add_yaxis(series_name, prepared_data, **filtered_series_opts)
                elif chart_type in ["pie", "gauge", "wordcloud"]:
                    data_pair = [(item['name'], item['value']) for item in series_values]
                    chart_instance.add(series_name, data_pair, **filtered_series_opts)
                elif chart_type == "geo":
                    data_pair = [(item['name'], item['value']) for item in series_values]
                    geo_type = s.get("type_", None)
                    if geo_type:
                        final_series_opts.pop("type_")
                    else:
                        geo_type = ChartType.SCATTER
                    chart_instance.add(series_name, data_pair, type_=geo_type, **filtered_series_opts)
                elif chart_type == "liquid":
                    chart_instance.add(series_name, series_values, **filtered_series_opts)
    
        def _render_chart(self, chart_to_render: Any, render_path: str) -> str:
            # output_dir = os.path.dirname(render_path)
            # if output_dir:
                # os.makedirs(output_dir, exist_ok=True)
            file_name = os.path.basename(render_path)
            render_path = f"/mnt/data/{file_name}"
            chart_to_render.render(render_path)
            print('html path:', render_path)
            return os.path.abspath(render_path)
    
    
        def create(
            self,
            chart_type: str,
            data: Dict,
            global_opts: Dict = None,
            series_opts: Dict = None,
            render_path: str = "chart.html",
            # is_base_chart: bool = True # 新增参数，标识是否为叠加图的基础图表
        ) -> Any: # 修改返回类型，如果is_base_chart为True，则返回图表实例
            
            final_global_opts = copy.deepcopy(self.DEFAULT_GLOBAL_OPTS)
            if global_opts:
                _deep_merge_dicts(final_global_opts, global_opts)
            
            series_opts_config = series_opts or {}
            
            chart_instance = self._initialize_chart(chart_type, final_global_opts.get("init", {}))
            self._apply_global_options(chart_instance, final_global_opts)
            self._add_series_data(chart_instance, chart_type, data, series_opts_config)
    
    
            return self._render_chart(chart_instance, render_path)
        
        """
