using Godot;
using System;
using System.Threading;
using Silk.NET.OpenCL;

[GlobalClass]
public partial class sim_mandelbrot : sim, IParameterized
{
	private const int MaxIterations = 10000;
	
	private float zoomLevel = 1.0f;
	private float centerX = -0.746f;
	private float centerY = 0.09f;
	
	public override string GetName() => "Mandelbrot";
	
	public void SetParameter(string name, object value)
	{
		switch (name.ToLower())
		{
			case "zoom":
				zoomLevel = Convert.ToSingle(value);
				break;
			case "centerx":
				centerX = Convert.ToSingle(value);
				break;
			case "centery":
				centerY = Convert.ToSingle(value);
				break;
		}
	}
	
	public object GetParameter(string name)
	{
		return name.ToLower() switch
		{
			"zoom" => zoomLevel,
			"centerx" => centerX,
			"centery" => centerY,
			_ => null
		};
	}
	
	public string[] GetParameterNames() => new[] { "zoom", "centerX", "centerY" };
	
	protected override void RenderCPU()
	{
		for (int t = 0; t < threadCount; t++)
		{
			int index = t;
			Thread thread = new Thread(() => RenderRows(index, threadCount));
			thread.IsBackground = false;
			thread.Start();
		}
	}
	
	private void RenderRows(int threadIndex, int totalThreads)
	{
		for (int py = threadIndex; py < height; py += totalThreads)
		{
			for (int px = 0; px < width; px++)
			{
				ProcessPixel(px, py);
			}
		}
		ThreadCompleted();
	}
	
	private void ProcessPixel(int px, int py)
	{
		float scale = width * zoomLevel;
		
		float x = centerX + (px - width * 0.5f) / scale;
		float y = centerY + (py - height * 0.5f) / scale;
		
		Color color = CalculateMandelbrot(x, y);
		SetPixelColor(px, py, color);
	}
	
	private Color CalculateMandelbrot(float x0, float y0)
	{
		float x = 0.0f;
		float y = 0.0f;
		
		for (int i = 1; i < MaxIterations; i++)
		{
			float magnitude = x * x + y * y;
			if (magnitude > 4.0f)
			{
				return IterationToColor(i);
			}
			
			float xtemp = x * x - y * y + x0;
			y = 2.0f * x * y + y0;
			x = xtemp;
		}
		
		return new Color(0, 0, 0);
	}
	
	private Color IterationToColor(int i)
	{
		float t = i / 255.0f;
		return new Color(
			t * 30 / 255.0f,
			t * 80 / 255.0f,
			(150 + t * 105) / 255.0f
		);
	}
	
	protected override string GetKernelName() => "mandelbrot";
	
	protected override void SetKernelArgs()
	{
		unsafe
		{
			int w = width;
			int h = height;
			float z = zoomLevel;
			nint buf = gpuBuffer;
			
			cl.SetKernelArg(gpuKernel, 0, (nuint)sizeof(nint), &buf);
			cl.SetKernelArg(gpuKernel, 1, (nuint)sizeof(int), &w);
			cl.SetKernelArg(gpuKernel, 2, (nuint)sizeof(int), &h);
			cl.SetKernelArg(gpuKernel, 3, (nuint)sizeof(float), &z);
		}
	}
	
	protected override string GetKernelSource() => @"
__kernel void mandelbrot(__global uchar* output, int width, int height, float zoom) {
	int index = get_global_id(0);
	if (index >= width * height) return;
	
	int px = index % width;
	int py = index / width;
	
	float scale = width * zoom;
	float x0 = -0.746f + (px - width * 0.5f) / scale;
	float y0 = 0.09f + (py - height * 0.5f) / scale;
	
	float x = 0.0f, y = 0.0f;
	int i;
	for (i = 1; i < 10000; i++) {
		float mag = x*x + y*y;
		if (mag > 4.0f) break;
		float xtemp = x*x - y*y + x0;
		y = 2.0f*x*y + y0;
		x = xtemp;
	}
	
	int outIndex = index * 3;
	if (i < 10000) {
		float t = (i / 255.0f);
		output[outIndex] = (uchar)(t * 30);
		output[outIndex + 1] = (uchar)(t * 80);
		output[outIndex + 2] = (uchar)(150 + t * 105);
	} else {
		output[outIndex] = 0;
		output[outIndex + 1] = 0;
		output[outIndex + 2] = 0;
	}
}";
}
