extends Node

@export var texture_rect: Node

var img: Image

var img_height = 512
var img_width = 1024

var max_thread_count

func _ready():
	mandelbrot.ChangeTextureSize(img_width, img_height)
	max_thread_count = OS.get_processor_count()
	mandelbrot.ChangeThreadCount(16)
	img = Image.create(img_width, img_height, false, Image.FORMAT_RGB8)

func _process(_delta: float):
	img.set_data(img_width, img_height, false, Image.FORMAT_RGB8, mandelbrot.GetData())
	var texture = ImageTexture.create_from_image(img)
	texture_rect.texture = texture
		
func do_mandelbrot():
	mandelbrot.MandelbrotTiled()
