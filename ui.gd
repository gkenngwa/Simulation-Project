extends Node

@export var texture_rect: Node
@export var max_cpu_cores_text: Label
@export var max_gpu_cores_text: Label
@export var h_slider: HSlider
@export var h_slider2: HSlider
@export var cpu_on_text: Label
@export var zoom_text: Label
@export var results_text: Label
@export var sim_selector: OptionButton

var img: Image

var img_height = 512
var img_width = 1024

var max_thread_count: int
var thread_count: int = 1

var max_work_group_count: int
var work_group_count: int = 1

var use_cpu: bool = true
var current_zoom: float = 1.0

var simulations = ["Mandelbrot", "Game of Life", "Monte Carlo"]

func _ready():
	for sim_name in simulations:
		sim_selector.add_item(sim_name)
	sim_selector.selected = 0
	sim_manager.Setsim("Mandelbrot")
	
	max_thread_count = OS.get_processor_count()
	max_work_group_count = sim_manager.GetMaxWorkGroupSize()
	
	h_slider.max_value = max_thread_count
	h_slider2.max_value = max_work_group_count
	
	cpu_slider_changed(1)
	gpu_slider_changed(1)
	
	img = Image.create(img_width, img_height, false, Image.FORMAT_RGB8)

func _process(_delta: float):
	if sim_manager.IsFinished():
		var data = sim_manager.GetData()
		img.set_data(img_width, img_height, false, Image.FORMAT_RGB8, data)
		var texture = ImageTexture.create_from_image(img)
		texture_rect.texture = texture
		
		var current_sim = simulations[sim_selector.selected]
		var mode = sim_manager.GetLastComputeMode()
		
		if current_sim == "Game of Life":
			var gens = sim_manager.GetGenerationsCompleted()
			var cells = sim_manager.GetLiveCellCount()
			if mode == 0:
				var threads = sim_manager.GetLastThreadCount()
				results_text.text = "%d thread(s): %d generations, %d live cells in 1s" % [threads, gens, cells]
			else:
				var groups = sim_manager.GetLastWorkGroupCount()
				results_text.text = "%d work group(s): %d generations, %d live cells in 1s" % [groups, gens, cells]
		else:
			var time_ms = sim_manager.GetTimeMs()
			if time_ms != 0:
				if mode == 0:
					var threads = sim_manager.GetLastThreadCount()
					results_text.text = "%d thread(s) took %dms" % [threads, time_ms]
				else:
					var groups = sim_manager.GetLastWorkGroupCount()
					results_text.text = "%d work group(s) took %dms" % [groups, time_ms]

func start_render():
	var mode = 0 if use_cpu else 1
	sim_manager.StartRender(img_width, img_height, thread_count, mode, work_group_count)

func cpu_slider_changed(value: float) -> void:
	thread_count = int(value)
	max_cpu_cores_text.text = "CPU Thread Count: %d (Max: %d)" % [thread_count, max_thread_count]

func gpu_slider_changed(value: float) -> void:
	work_group_count = int(value)
	max_gpu_cores_text.text = "GPU Work Group Count: %d (Max: %d)" % [work_group_count, max_work_group_count]

func _on_check_button_toggled(toggled_on: bool) -> void:
	use_cpu = toggled_on
	cpu_on_text.text = "CPU" if toggled_on else "GPU"

func zoom_in():
	current_zoom *= 2.0
	_update_zoom()

func zoom_out():
	if current_zoom > 1.0:
		current_zoom /= 2.0
		_update_zoom()

func _update_zoom():
	zoom_text.text = "Zoom: %dx" % [int(current_zoom)]
	sim_manager.SetZoom(current_zoom)

func _on_sim_selector_item_selected(index: int) -> void:
	var sim_name = simulations[index]
	sim_manager.Setsim(sim_name)
	current_zoom = 1.0
	_update_zoom()
