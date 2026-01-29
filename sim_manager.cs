using Godot;
using System;
using System.Collections.Generic;

[GlobalClass]
public partial class sim_manager : RefCounted
{
	private static Dictionary<string, Func<sim>> registeredsims = new Dictionary<string, Func<sim>>();
	private static sim currentsim;
	private static string currentsimName;
	
	private static bool initialized = false;
	
	private static void EnsureInitialized()
	{
		if (!initialized)
		{
			Register("Mandelbrot", () => new sim_mandelbrot());
			Register("Game of Life", () => new sim_game_of_life());
			Register("Monte Carlo", () => new sim_monte_carlo());
			initialized = true;
		}
	}
	
	public static void Register(string name, Func<sim> factory)
	{
		registeredsims[name] = factory;
	}
	
	public static string[] GetAvailablesims()
	{
		var names = new string[registeredsims.Count];
		registeredsims.Keys.CopyTo(names, 0);
		return names;
	}
	
	public static void Setsim(string name)
	{
		EnsureInitialized();
		if (registeredsims.TryGetValue(name, out var factory))
		{
			currentsim = factory();
			currentsimName = name;
		}
		else
		{
			GD.PrintErr($"sim '{name}' not found");
		}
	}
	
	public static sim GetCurrentsim() => currentsim;
	public static string GetCurrentsimName() => currentsimName;
	
	public static bool IsFinished() => currentsim?.IsFinished() ?? false;
	public static bool IsRendering() => currentsim?.IsRendering() ?? false;
	public static float GetTimeMs() => currentsim?.GetTimeMs() ?? 0f;
	public static int GetLastThreadCount() => currentsim?.GetLastThreadCount() ?? 0;
	public static int GetLastWorkGroupCount() => currentsim?.GetLastWorkGroupCount() ?? 0;
	public static ComputeMode GetLastComputeMode() => currentsim?.GetLastComputeMode() ?? ComputeMode.CPU;
	public static byte[] GetData() => currentsim?.GetData();
	public static int GetMaxWorkGroupSize() => currentsim?.GetMaxWorkGroupSize() ?? 1;
	
	public static void StartRender(int width, int height, int threadCount, ComputeMode mode, int workGroupCount)
	{
		currentsim?.StartRender(width, height, threadCount, mode, workGroupCount);
	}
	
	public static void SetZoom(float zoom)
	{
		SetParameter("zoom", zoom);
	}
	
	public static int GetGenerationsCompleted()
	{
		return currentsim.GetGenerationsCompleted();
	}
	
	public static int GetLiveCellCount()
	{
		return currentsim.GetLiveCellCount();
	}
	
	public static void SetParameter(string name, object value)
	{
		if (currentsim is IParameterized paramSim)
		{
			paramSim.SetParameter(name, value);
		}
	}
	
	public static object GetParameter(string name)
	{
		if (currentsim is IParameterized paramSim)
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
