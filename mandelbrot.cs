using Godot;
using System;
using System.Threading;
using System.Diagnostics;
using Silk.NET.OpenCL;

[GlobalClass]
public partial class mandelbrot : RefCounted
{
	private const int maxIterations = 10000;
	private static int width, height;
	
	private static byte[] data;
	private static int threadCount;
	private static int workGroupCount;
	
	private static float zoomLevel = 1.0f;
	
	private static int mandelbrotType = 0;
	
	private static bool isRendering = false;
	private static int threadsCompleted = 0;
	private static int lastType = 0;
	
	private static Stopwatch sw = new Stopwatch();
	private static float lastTime = 0f;
	
	private static bool isFinished = false;
	
	public static bool IsFinished() {
		return isFinished;
	}
	
	public static float GetTime() {
		return lastTime;
	}
	
	public static int GetLastThreadCount() {
		return threadCount;
	}
	
	public static int GetLastMandelbrotType() {
		return mandelbrotType;
	}
	
	public static int GetLastWorkGroupCount() {
		return workGroupCount;
	}
	
	public static int GetMaxWorkGroupSize()
	{
		unsafe
		{
			var cl = CL.GetApi();
			
			uint platformCount = 0;
			cl.GetPlatformIDs(0, null, &platformCount);
			
			Span<nint> platforms = new nint[(int)platformCount];
			fixed (nint* pPlatforms = platforms)
			{
				cl.GetPlatformIDs(platformCount, pPlatforms, null);
				
				uint deviceCount = 0;
				cl.GetDeviceIDs(platforms[0], DeviceType.All, 0, null, &deviceCount);
				
				Span<nint> devices = new nint[(int)deviceCount];
				fixed (nint* pDevices = devices)
				{
					cl.GetDeviceIDs(platforms[0], DeviceType.All, deviceCount, pDevices, null);
					
					ulong workGroupSize = 0;
					cl.GetDeviceInfo(devices[0], DeviceInfo.MaxWorkGroupSize, sizeof(ulong), &workGroupSize, null);
					return (int)workGroupSize;
				}
			}
		}
	}
	
	public static void SetNewZoom(int newZoom) {
		zoomLevel = newZoom;
	}

	public static void StartMandelbrotRender(int newWidth, int newHeight, int newThreadCount, int newMandelbrotType, int newWorkGroupCount)
	{
		if (isRendering)
			return;
		
		isFinished = false;
		isRendering = true;
		threadsCompleted = 0;
		lastTime = 0f;
		width = newWidth;
		height = newHeight;
		threadCount = newThreadCount;
		workGroupCount = newWorkGroupCount;
		mandelbrotType = newMandelbrotType;
		data = new byte[newWidth * newHeight * 3];
		
		sw.Start();
		
		if (mandelbrotType == 0) {
			Mandelbrot();
		}
		else if (mandelbrotType == 1) {
			MandelbrotGPU();
		}
	}
		
	public static byte[] GetData() {
		return data;
	}
	
	public static void Mandelbrot()
	{
		int tileWidth = width / threadCount;
		
		for (int t = 0; t < threadCount; t++)
		{
			int index = t;
			int startX = index * tileWidth;
			int endX = (index == threadCount - 1) ? width : startX + tileWidth;
			
			Thread thread = null;
			
			thread = new Thread(() => RenderMandelbrot(index, threadCount));
				
			thread.IsBackground = false;
			thread.Start();
		}
	}
	
	public static void MandelbrotGPU()
	{
		if (!isGPUInitialized)
		{
			InitializeGPU();
		}
		
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
			
			nuint globalWorkSize = (nuint)(width * height);
			nuint localWorkSize = (nuint)workGroupCount;
			
			cl.EnqueueNdrangeKernel(gpuQueue, gpuKernel, 1, null, &globalWorkSize, &localWorkSize, 0, null, null);
			cl.Finish(gpuQueue);
			
			fixed (byte* pData = data)
			{
				cl.EnqueueReadBuffer(gpuQueue, gpuBuffer, true, 0, (nuint)(width * height * 3), pData, 0, null, null);
			}
			
			FinishMandelbrot();
		}
	}

	private static bool isGPUInitialized = false;
	private static nint gpuContext;
	private static nint gpuQueue;
	private static nint gpuProgram;
	private static nint gpuKernel;
	private static nint gpuBuffer;
	private static nint gpuDevice;
	private static CL cl;

	private static void InitializeGPU()
{
	unsafe
	{
		cl = CL.GetApi();
		
		nint platform;
		nint device;
		cl.GetPlatformIDs(1, &platform, null);
		cl.GetDeviceIDs(platform, DeviceType.Gpu, 1, &device, null);
		gpuDevice = device;

		int err;
		nint dev = gpuDevice;
		gpuContext = cl.CreateContext(null, 1, &dev, null, null, &err);
		gpuQueue = cl.CreateCommandQueue(gpuContext, gpuDevice, CommandQueueProperties.None, &err);
		
		string kernelSource = @"
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
		
		byte[] sourceBytes = System.Text.Encoding.ASCII.GetBytes(kernelSource);
		nuint sourceLength = (nuint)sourceBytes.Length;
		
		nint prog;
		nint kern;
		nint buf;
		
		prog = cl.CreateProgramWithSource(gpuContext, 1, new string[] { kernelSource }, null, &err);

		cl.BuildProgram(prog, 1, &dev, "", null, null);

		byte[] kernelName = System.Text.Encoding.ASCII.GetBytes("mandelbrot\0");
		fixed (byte* pKernelName = kernelName)
		{
			kern = cl.CreateKernel(prog, pKernelName, &err);
		}

		buf = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)(width * height * 3), null, &err);
		
		gpuProgram = prog;
		gpuKernel = kern;
		gpuBuffer = buf;
		
		isGPUInitialized = true;
	}
}

	private static void RenderMandelbrotTile(int startX, int endX)
	{
		for (int px = startX; px < endX; px++)
		{
			for (int py = 0; py < height; py++)
			{
				ProcessPixel(px, py);
			}
		}
		
		ThreadCompleted();
	}
	
	private static void RenderMandelbrot(int threadIndex, int threadCount)
	{
		for (int py = threadIndex; py < height; py += threadCount)
		{
			for (int px = 0; px < width; px++)
			{
				ProcessPixel(px, py);
			}
		}
		
		ThreadCompleted();
	}
	
	private static void ThreadCompleted() {
		threadsCompleted++;
		if (threadsCompleted == threadCount)
		{
			FinishMandelbrot();
		}
	}
	
	private static void FinishMandelbrot() {
		isRendering = false;
		isFinished = true;
		sw.Stop();
		lastTime = sw.ElapsedMilliseconds;
		sw.Reset();
	}
	
	private static void ProcessPixel(int px, int py)
	{
		float scale = width * zoomLevel;
		
		float centerX = -0.746f;
		float centerY = 0.09f;
				
		float x = centerX + (px - width * 0.5f) / scale;
		float y = centerY + (py - height * 0.5f) / scale;
		
		Color color = CalculateMandelbrot(x, y);
		
		SetPixelColor(px, py, color);
	}
	
	private static Color CalculateMandelbrot(float x0, float y0)
	{
		float x = 0.0f;
		float y = 0.0f;
		
		for (int i = 1; i < maxIterations; i++)
		{
			float magnitude = x * x + y * y;
			if (magnitude > 4.0f)
			{
				return RgbConv(i);
			}
			
			float xtemp = x * x - y * y + x0;
			y = 2.0f * x * y + y0;
			x = xtemp;
		}
		
		return new Color(0, 0, 0);
	}
	
	private static void SetPixelColor(int px, int py, Color color)
	{
		int index = (py * width + px) * 3;
		data[index] = (byte)(color.R * 255);
		data[index + 1] = (byte)(color.G * 255);
		data[index + 2] = (byte)(color.B * 255);
	}
	
	private static Color RgbConv(int i)
	{
		float t = (i / 255.0f);
		
		return new Color(
			t * 30 / 255.0f,
			t * 80 / 255.0f,
			(150 + t * 105) / 255.0f
		);
	}
}
