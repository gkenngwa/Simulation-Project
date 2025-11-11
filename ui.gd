extends Node

@export var texture_rect: Node
@export var max_cpu_cores_text : Label
@export var max_gpu_cores_text : Label
@export var h_slider: HSlider
@export var h_slider2: HSlider
@export var cpu_on_text : Label
@export var zoom_text : Label
@export var results_text : Label

var img: Image

var img_height = 512
var img_width = 1024

var max_thread_count
var thread_count = 1

var max_work_group_count
var work_group_count = 1

var cpu_on = true

var current_zoom = 1.0

func _ready():
	max_thread_count = OS.get_processor_count()
	max_work_group_count = mandelbrot.GetMaxWorkGroupSize()
	h_slider.max_value = max_thread_count
	h_slider2.max_value = max_work_group_count
	cpu_slider_changed(1)
	gpu_slider_changed(1)
	img = Image.create(img_width, img_height, false, Image.FORMAT_RGB8)
	
func _process(_delta: float):
	if mandelbrot.IsFinished():
		var data = mandelbrot.GetData()
		img.set_data(img_width, img_height, false, Image.FORMAT_RGB8, data)
		var texture = ImageTexture.create_from_image(img)
		texture_rect.texture = texture
		if mandelbrot.GetTime() != 0:
			if mandelbrot.GetLastMandelbrotType() == 0:
				results_text.text = "%d thread(s) took %dms" % [mandelbrot.GetLastThreadCount(), mandelbrot.GetTime()]
			elif mandelbrot.GetLastMandelbrotType() == 1:
				results_text.text = "%d work group(s) took %dms" % [mandelbrot.GetLastWorkGroupCount(), mandelbrot.GetTime()]
	
func do_mandelbrot():
	if cpu_on:
		mandelbrot.StartMandelbrotRender(img_width, img_height, thread_count, 0, work_group_count)
	else:
		mandelbrot.StartMandelbrotRender(img_width, img_height, thread_count, 1, work_group_count)
	
func cpu_slider_changed(value: float) -> void:
	thread_count = value
	max_cpu_cores_text.text = "CPU Thread Count: %d (Max: %d)" % [thread_count, max_thread_count]
	
func gpu_slider_changed(value: float) -> void:
	work_group_count = value
	max_gpu_cores_text.text = "GPU Work Group Count: %d (Max: %d)" % [work_group_count, max_work_group_count]

func _on_check_button_toggled(toggled_on: bool) -> void:
	cpu_on = toggled_on
	cpu_on_text.text = "CPU" if toggled_on else "GPU"
	
func zoom_in():
	current_zoom = current_zoom * 2.0
	_update_zoom()
	
func zoom_out():
	if current_zoom > 1.0:
		current_zoom = current_zoom / 2.0
		_update_zoom()

func _update_zoom():
	zoom_text.text = "Zoom: %dx" % [current_zoom]
	mandelbrot.SetNewZoom(current_zoom)
