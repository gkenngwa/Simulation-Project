using Godot;
using System;
using System.Diagnostics;
using System.Threading;
using Silk.NET.OpenCL;

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
	protected int threadsCompleted = 0;
	
	protected Stopwatch stopwatch = new Stopwatch();
	protected float lastTimeMs = 0f;
	
	protected int width;
	protected int height;
	protected int threadCount;
	protected int workGroupCount;
	protected ComputeMode lastComputeMode;
	
	protected byte[] data;
	
	protected bool isGPUInitialized = false;
	protected nint gpuContext;
	protected nint gpuQueue;
	protected nint gpuProgram;
	protected nint gpuKernel;
	protected nint gpuBuffer;
	protected nint gpuDevice;
	protected nint gpuGridA;
	protected nint gpuGridB;
	protected CL cl;
	
	protected int gridSize = 0;
	protected byte[] gridA;
	protected byte[] gridB;
	protected Barrier barrier;
	protected bool timeUp = false;
	
	public abstract string GetName();
	protected abstract void RenderCPU();
	protected abstract string GetKernelSource();
	protected abstract string GetKernelName();
	protected abstract void SetKernelArgs();
	
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
	
	public int GetMaxWorkGroupSize()
	{
		unsafe
		{
			var clApi = CL.GetApi();
			uint platformCount = 0;
			clApi.GetPlatformIDs(0, null, &platformCount);
			Span<nint> platforms = new nint[(int)platformCount];
			fixed (nint* pPlatforms = platforms)
			{
				clApi.GetPlatformIDs(platformCount, pPlatforms, null);
				uint deviceCount = 0;
				clApi.GetDeviceIDs(platforms[0], DeviceType.All, 0, null, &deviceCount);
				Span<nint> devices = new nint[(int)deviceCount];
				fixed (nint* pDevices = devices)
				{
					clApi.GetDeviceIDs(platforms[0], DeviceType.All, deviceCount, pDevices, null);
					ulong workGroupSize = 0;
					clApi.GetDeviceInfo(devices[0], DeviceInfo.MaxWorkGroupSize, sizeof(ulong), &workGroupSize, null);
					return (int)workGroupSize;
				}
			}
		}
	}
	
	public void StartRender(int newWidth, int newHeight, int newThreadCount, ComputeMode mode, int newWorkGroupCount)
	{
		if (isRendering) return;
		
		isFinished = false;
		isRendering = true;
		threadsCompleted = 0;
		lastTimeMs = 0f;
		
		width = newWidth;
		height = newHeight;
		threadCount = newThreadCount;
		workGroupCount = newWorkGroupCount;
		lastComputeMode = mode;
		data = new byte[newWidth * newHeight * 3];
		
		stopwatch.Start();
		
		if (mode == ComputeMode.CPU)
			RenderCPU();
		else
			RenderGPU();
	}
	
	protected void InitializeGrids(int size)
	{
		gridSize = size;
		gridA = new byte[size * size];
		gridB = new byte[size * size];
	}
	
	protected void SwapGrids()
	{
		var tmp = gridA;
		gridA = gridB;
		gridB = tmp;
	}
	
	protected void StartTimedSimulation(Action<int, int> threadWork)
	{
		timeUp = false;
		barrier = new Barrier(threadCount);
		
		new Thread(() => { Thread.Sleep(1000); timeUp = true; }) { IsBackground = true }.Start();
		
		for (int t = 0; t < threadCount; t++)
		{
			int idx = t;
			new Thread(() => threadWork(idx, threadCount)) { IsBackground = false }.Start();
		}
	}
	
	protected virtual void RenderGPU()
	{
		if (!isGPUInitialized)
			InitializeGPU();
		
		unsafe
		{
			SetKernelArgs();
			
			nuint globalWorkSize = (nuint)(width * height);
			nuint localWorkSize = (nuint)workGroupCount;
			
			cl.EnqueueNdrangeKernel(gpuQueue, gpuKernel, 1, null, &globalWorkSize, &localWorkSize, 0, null, null);
			cl.Finish(gpuQueue);
			
			fixed (byte* pData = data)
			{
				cl.EnqueueReadBuffer(gpuQueue, gpuBuffer, true, 0, (nuint)(width * height * 3), pData, 0, null, null);
			}
			
			FinishRender();
		}
	}
	
	protected void InitializeGPU()
	{
		unsafe
		{
			cl = CL.GetApi();
			
			nint platform, device;
			cl.GetPlatformIDs(1, &platform, null);
			cl.GetDeviceIDs(platform, DeviceType.Gpu, 1, &device, null);
			gpuDevice = device;
			
			int err;
			nint dev = gpuDevice;
			gpuContext = cl.CreateContext(null, 1, &dev, null, null, &err);
			gpuQueue = cl.CreateCommandQueue(gpuContext, gpuDevice, CommandQueueProperties.None, &err);
			
			string kernelSource = GetKernelSource();
			nint prog = cl.CreateProgramWithSource(gpuContext, 1, new string[] { kernelSource }, null, &err);
			cl.BuildProgram(prog, 1, &dev, "", null, null);
			
			byte[] kernelName = System.Text.Encoding.ASCII.GetBytes(GetKernelName() + "\0");
			fixed (byte* pKernelName = kernelName)
			{
				gpuKernel = cl.CreateKernel(prog, pKernelName, &err);
			}
			
			gpuProgram = prog;
			gpuBuffer = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)(width * height * 3), null, &err);
			
			isGPUInitialized = true;
		}
	}
	
	protected void InitializeGPUGrids(int size)
	{
		unsafe
		{
			if (!isGPUInitialized) InitializeGPU();
			int err;
			gpuGridA = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)(size * size), null, &err);
			gpuGridB = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)(size * size), null, &err);
		}
	}
	
	protected void ThreadCompleted()
	{
		threadsCompleted++;
		if (threadsCompleted == threadCount)
			FinishRender();
	}
	
	protected void FinishRender()
	{
		isRendering = false;
		isFinished = true;
		stopwatch.Stop();
		lastTimeMs = stopwatch.ElapsedMilliseconds;
		stopwatch.Reset();
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
}
