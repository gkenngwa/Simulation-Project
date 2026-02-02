using Godot;
using System;
using System.Diagnostics;

public enum ComputeMode
{
	CPU,
	GPU
}

[GlobalClass]
public abstract partial class sim : RefCounted
{
	protected bool isRendering = false;
	protected bool isFinished = false;
	
	protected Stopwatch stopwatch = new Stopwatch();
	protected float lastTimeMs = 0f;
	
	protected int width;
	protected int height;
	protected int threadCount;
	protected int workGroupCount;
	protected ComputeMode lastComputeMode;
	
	protected byte[] data;
	
	protected int gridSize = 0;
	protected byte[] gridA;
	protected byte[] gridB;
	
	public abstract string GetName();
	public abstract string GetKernelSource();
	public abstract string GetKernelName();
	
	public virtual GpuArg[] GetKernelArgs() => null;
	public virtual void OnGPUSetup(gpu_handler gpu) { }
	public virtual void RunGPUCompute(gpu_handler gpu, Action onComplete) => DefaultGPURender(gpu, onComplete);
	public virtual void RunCPUCompute(cpu_handler cpu, Action onComplete) => DefaultCPURender(cpu, onComplete);
	public virtual void ComputePixel(int px, int py) { }
	
	public bool IsFinished() => isFinished;
	public bool IsRendering() => isRendering;
	public float GetTimeMs() => lastTimeMs;
	public int GetLastThreadCount() => threadCount;
	public int GetLastWorkGroupCount() => workGroupCount;
	public ComputeMode GetLastComputeMode() => lastComputeMode;
	public byte[] GetData() => data;
	public int GetWidth() => width;
	public int GetHeight() => height;
	public virtual int GetGenerationsCompleted() => 0;
	public virtual int GetLiveCellCount() => 0;
	
	public void Setup(int newWidth, int newHeight, int newThreadCount, int newWorkGroupCount, ComputeMode mode)
	{
		isFinished = false;
		isRendering = true;
		lastTimeMs = 0f;
		
		width = newWidth;
		height = newHeight;
		threadCount = newThreadCount;
		workGroupCount = newWorkGroupCount;
		lastComputeMode = mode;
		data = new byte[newWidth * newHeight * 3];
		
		stopwatch.Restart();
	}
	
	public void FinishRender()
	{
		isRendering = false;
		isFinished = true;
		stopwatch.Stop();
		lastTimeMs = stopwatch.ElapsedMilliseconds;
	}
	
	protected void InitializeGrids(int size)
	{
		gridSize = size;
		gridA = new byte[size * size];
		gridB = new byte[size * size];
	}
	
	protected void SwapGrids()
	{
		(gridA, gridB) = (gridB, gridA);
	}
	
	protected void SetPixelColor(int px, int py, byte r, byte g, byte b)
	{
		int index = (py * width + px) * 3;
		data[index] = r;
		data[index + 1] = g;
		data[index + 2] = b;
	}
	
	protected void SetPixelColor(int px, int py, Color color)
	{
		SetPixelColor(px, py, (byte)(color.R * 255), (byte)(color.G * 255), (byte)(color.B * 255));
	}
	
	private void DefaultCPURender(cpu_handler cpu, Action onComplete)
	{
		cpu.RunParallel(threadCount, (idx, total) =>
		{
			for (int py = idx; py < height; py += total)
			{
				for (int px = 0; px < width; px++)
				{
					ComputePixel(px, py);
				}
			}
		}, onComplete);
	}
	
	private void DefaultGPURender(gpu_handler gpu, Action onComplete)
	{
		gpu.RunKernel(GetKernelName(), width * height, workGroupCount, GetKernelArgs());
		gpu.ReadOutputBuffer(data);
		onComplete();
	}
}
