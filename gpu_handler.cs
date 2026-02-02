using Godot;
using System;
using System.Collections.Generic;
using Silk.NET.OpenCL;

public enum GpuArgType { Buffer, GridA, GridB, Int, Float }

public struct GpuArg
{
	public GpuArgType Type;
	public object Value;
	
	public static GpuArg Buffer => new GpuArg { Type = GpuArgType.Buffer };
	public static GpuArg GridA => new GpuArg { Type = GpuArgType.GridA };
	public static GpuArg GridB => new GpuArg { Type = GpuArgType.GridB };
	public static GpuArg Int(int v) => new GpuArg { Type = GpuArgType.Int, Value = v };
	public static GpuArg Float(float v) => new GpuArg { Type = GpuArgType.Float, Value = v };
}

[GlobalClass]
public partial class gpu_handler : RefCounted
{
	private CL cl;
	private nint gpuContext;
	private nint gpuQueue;
	private nint gpuDevice;
	private nint gpuProgram;
	private nint gpuBuffer;
	private nint gpuGridA;
	private nint gpuGridB;
	private Dictionary<string, nint> kernels = new Dictionary<string, nint>();
	
	private bool isInitialized = false;
	private int maxWorkGroupSize = 0;
	
	public bool IsInitialized => isInitialized;
	
	public int GetMaxWorkGroupSize()
	{
		if (maxWorkGroupSize > 0) return maxWorkGroupSize;
		
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
					maxWorkGroupSize = (int)workGroupSize;
					return maxWorkGroupSize;
				}
			}
		}
	}
	
	public void Initialize()
	{
		if (isInitialized) return;
		
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
			
			isInitialized = true;
		}
	}
	
	public void BuildKernel(string kernelSource, string kernelName)
	{
		if (!isInitialized) Initialize();
		
		unsafe
		{
			int err;
			nint dev = gpuDevice;
			nint prog = cl.CreateProgramWithSource(gpuContext, 1, new string[] { kernelSource }, null, &err);
			cl.BuildProgram(prog, 1, &dev, "", null, null);
			gpuProgram = prog;
			
			AddKernel(kernelName);
		}
	}
	
	public void AddKernel(string kernelName)
	{
		if (kernels.ContainsKey(kernelName)) return;
		
		unsafe
		{
			int err;
			byte[] kernelNameBytes = System.Text.Encoding.ASCII.GetBytes(kernelName + "\0");
			fixed (byte* pKernelName = kernelNameBytes)
			{
				kernels[kernelName] = cl.CreateKernel(gpuProgram, pKernelName, &err);
			}
		}
	}
	
	public nint GetKernel(string kernelName)
	{
		if (kernels.TryGetValue(kernelName, out var kernel))
			return kernel;
		
		AddKernel(kernelName);
		return kernels[kernelName];
	}
	
	public void CreateBuffer(int size)
	{
		if (!isInitialized) Initialize();
		
		unsafe
		{
			int err;
			gpuBuffer = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)size, null, &err);
		}
	}
	
	public void CreateGridBuffers(int size)
	{
		if (!isInitialized) Initialize();
		
		unsafe
		{
			int err;
			gpuGridA = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)size, null, &err);
			gpuGridB = cl.CreateBuffer(gpuContext, MemFlags.ReadWrite, (nuint)size, null, &err);
		}
	}
	
	public void SetKernelArgs(nint kernel, params GpuArg[] args)
	{
		for (uint i = 0; i < args.Length; i++)
		{
			var arg = args[i];
			switch (arg.Type)
			{
				case GpuArgType.Buffer:
					SetKernelArgBuffer(kernel, i, gpuBuffer);
					break;
				case GpuArgType.GridA:
					SetKernelArgBuffer(kernel, i, gpuGridA);
					break;
				case GpuArgType.GridB:
					SetKernelArgBuffer(kernel, i, gpuGridB);
					break;
				case GpuArgType.Int:
					SetKernelArgInt(kernel, i, (int)arg.Value);
					break;
				case GpuArgType.Float:
					SetKernelArgFloat(kernel, i, (float)arg.Value);
					break;
			}
		}
	}
	
	private void SetKernelArgInt(nint kernel, uint index, int value)
	{
		unsafe
		{
			cl.SetKernelArg(kernel, index, (nuint)sizeof(int), &value);
		}
	}
	
	private void SetKernelArgFloat(nint kernel, uint index, float value)
	{
		unsafe
		{
			cl.SetKernelArg(kernel, index, (nuint)sizeof(float), &value);
		}
	}
	
	private void SetKernelArgBuffer(nint kernel, uint index, nint buffer)
	{
		unsafe
		{
			cl.SetKernelArg(kernel, index, (nuint)sizeof(nint), &buffer);
		}
	}
	
	public void EnqueueKernel(nint kernel, int globalSize, int localSize)
	{
		unsafe
		{
			nuint gs = (nuint)globalSize;
			nuint ls = (nuint)localSize;
			cl.EnqueueNdrangeKernel(gpuQueue, kernel, 1, null, &gs, &ls, 0, null, null);
		}
	}
	
	public void Finish()
	{
		cl.Finish(gpuQueue);
	}
	
	public void WriteBuffer(nint buffer, byte[] data)
	{
		unsafe
		{
			fixed (byte* p = data)
			{
				cl.EnqueueWriteBuffer(gpuQueue, buffer, true, 0, (nuint)data.Length, p, 0, null, null);
			}
		}
	}
	
	public void ReadBuffer(nint buffer, byte[] data)
	{
		unsafe
		{
			fixed (byte* p = data)
			{
				cl.EnqueueReadBuffer(gpuQueue, buffer, true, 0, (nuint)data.Length, p, 0, null, null);
			}
		}
	}
	
	public void SwapGrids()
	{
		(gpuGridA, gpuGridB) = (gpuGridB, gpuGridA);
	}
	
	public int RunTimedKernelLoop(string kernelName, int globalSize, int localSize, int durationMs, Func<GpuArg[]> getArgs)
	{
		var kernel = GetKernel(kernelName);
		int iterations = 0;
		var start = DateTime.Now;
		
		while ((DateTime.Now - start).TotalMilliseconds < durationMs)
		{
			SetKernelArgs(kernel, getArgs());
			EnqueueKernel(kernel, globalSize, localSize);
			Finish();
			SwapGrids();
			iterations++;
		}
		
		return iterations;
	}
	
	public void RunKernel(string kernelName, int globalSize, int localSize, GpuArg[] args)
	{
		var kernel = GetKernel(kernelName);
		SetKernelArgs(kernel, args);
		EnqueueKernel(kernel, globalSize, localSize);
		Finish();
	}
	
	public void WriteGridA(byte[] data)
	{
		WriteBuffer(gpuGridA, data);
	}
	
	public void ReadGridA(byte[] data)
	{
		ReadBuffer(gpuGridA, data);
	}
	
	public void ReadOutputBuffer(byte[] data)
	{
		ReadBuffer(gpuBuffer, data);
	}
}
