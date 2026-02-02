using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class sim_manager : RefCounted
{
	private static Dictionary<string, Func<sim>> registeredSims = new Dictionary<string, Func<sim>>();
	private static sim currentSim;
	private static string currentSimName;
	
	private static gpu_handler gpuHandler;
	private static cpu_handler cpuHandler;
	
	private static bool initialized = false;
	private static bool gpuSetupForCurrentSim = false;
	
	private static void EnsureInitialized()
	{
		if (!initialized)
		{
			Register("Mandelbrot", () => new sim_mandelbrot());
			Register("Game of Life", () => new sim_game_of_life());
			Register("Monte Carlo", () => new sim_monte_carlo());
			
			gpuHandler = new gpu_handler();
			gpuHandler.Initialize();
			cpuHandler = new cpu_handler();
			
			initialized = true;
		}
	}
	
	public static void Register(string name, Func<sim> factory)
	{
		registeredSims[name] = factory;
	}
	
	public static string[] GetAvailableSims()
	{
		var names = new string[registeredSims.Count];
		registeredSims.Keys.CopyTo(names, 0);
		return names;
	}
	
	public static void Setsim(string name)
	{
		EnsureInitialized();
		if (registeredSims.TryGetValue(name, out var factory))
		{
			currentSim = factory();
			currentSimName = name;
			gpuSetupForCurrentSim = false;
		}
		else
		{
			GD.PrintErr($"Sim '{name}' not found");
		}
	}
	
	public static sim GetCurrentSim() => currentSim;
	public static string GetCurrentSimName() => currentSimName;
	
	public static bool IsFinished() => currentSim?.IsFinished() ?? false;
	public static bool IsRendering() => currentSim?.IsRendering() ?? false;
	public static float GetTimeMs() => currentSim?.GetTimeMs() ?? 0f;
	public static int GetLastThreadCount() => currentSim?.GetLastThreadCount() ?? 0;
	public static int GetLastWorkGroupCount() => currentSim?.GetLastWorkGroupCount() ?? 0;
	public static ComputeMode GetLastComputeMode() => currentSim?.GetLastComputeMode() ?? ComputeMode.CPU;
	public static byte[] GetData() => currentSim?.GetData();
	public static int GetMaxWorkGroupSize() => gpuHandler?.GetMaxWorkGroupSize() ?? 1;
	
	public static void StartRender(int width, int height, int threadCount, ComputeMode mode, int workGroupCount)
	{
		EnsureInitialized();
		if (currentSim == null || currentSim.IsRendering()) return;
		
		currentSim.Setup(width, height, threadCount, workGroupCount, mode);
		
		if (mode == ComputeMode.CPU)
		{
			currentSim.RunCPUCompute(cpuHandler, () => currentSim.FinishRender());
		}
		else
		{
			SetupGPUForSim(width, height);
			currentSim.RunGPUCompute(gpuHandler, () => currentSim.FinishRender());
		}
	}
	
	private static void SetupGPUForSim(int width, int height)
	{
		if (!gpuSetupForCurrentSim)
		{
			gpuHandler.BuildKernel(currentSim.GetKernelSource(), currentSim.GetKernelName());
			gpuHandler.CreateBuffer(width * height * 3);
			currentSim.OnGPUSetup(gpuHandler);
			gpuSetupForCurrentSim = true;
		}
	}
	
	public static void SetZoom(float zoom)
	{
		SetParameter("zoom", zoom);
	}
	
	public static int GetGenerationsCompleted()
	{
		return currentSim?.GetGenerationsCompleted() ?? 0;
	}
	
	public static int GetLiveCellCount()
	{
		return currentSim?.GetLiveCellCount() ?? 0;
	}
	
	public static void SetParameter(string name, object value)
	{
		if (currentSim is IParameterized paramSim)
		{
			paramSim.SetParameter(name, value);
		}
	}
	
	public static object GetParameter(string name)
	{
		if (currentSim is IParameterized paramSim)
		{
			return paramSim.GetParameter(name);
		}
		return null;
	}
}

public interface IParameterized
{
	void SetParameter(string name, object value);
	object GetParameter(string name);
	string[] GetParameterNames();
}
