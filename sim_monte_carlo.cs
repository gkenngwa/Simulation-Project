using Godot;
using System;

[GlobalClass]
public partial class sim_monte_carlo : sim, IParameterized
{
	private int samplesPerRay = 32;
	private float zoom = 1.0f;
	
	private uint threadSeed;
	private float threadTint;

	public override string GetName() => "Monte Carlo";

	public void SetParameter(string name, object value)
	{
		switch (name.ToLower())
		{
			case "samplesperray": samplesPerRay = Convert.ToInt32(value); break;
			case "zoom": zoom = Convert.ToSingle(value); break;
		}
	}

	public object GetParameter(string name) => name.ToLower() switch
	{
		"samplesperray" => samplesPerRay,
		"zoom" => zoom,
		_ => null
	};
	
	public string[] GetParameterNames() => new[] { "samplesPerRay", "zoom" };
	
	public override void RunCPUCompute(cpu_handler cpu, Action onComplete)
	{
		cpu.RunParallel(threadCount, (idx, total) =>
		{
			uint seed = (uint)(idx * 12345 + 1);
			float tint = idx / (float)total;
			
			for (int py = idx; py < height; py += total)
			{
				for (int px = 0; px < width; px++)
				{
					ComputePixelWithSeed(px, py, ref seed, tint);
				}
			}
		}, onComplete);
	}
	
	public override void ComputePixel(int px, int py)
	{
		ComputePixelWithSeed(px, py, ref threadSeed, threadTint);
	}
	
	private void ComputePixelWithSeed(int px, int py, ref uint seed, float tint)
	{
		float sum = 0;
		for (int j = 0; j < samplesPerRay; j++)
		{
			float rx = px + Rf(ref seed) - 0.5f;
			float ry = py + Rf(ref seed) - 0.5f;
			sum += ((int)rx ^ (int)ry) & 255;
		}
		byte c = (byte)(sum / samplesPerRay);
		byte r = (byte)(c * (0.5f + 0.5f * tint));
		byte g = (byte)(c * (1f - 0.5f * tint));
		byte b = (byte)(c * 0.8f);
		SetPixelColor(px, py, r, g, b);
	}

	private static uint Rng(uint s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
	private static float Rf(ref uint s) { s = Rng(s); return (s & 0xFFFFFF) / (float)0xFFFFFF; }

	public override string GetKernelName() => "monte_carlo_volume";
	
	public override GpuArg[] GetKernelArgs() => new[]
	{
		GpuArg.Buffer,
		GpuArg.Int(width),
		GpuArg.Int(height),
		GpuArg.Int(samplesPerRay),
		GpuArg.Float(zoom)
	};

	public override string GetKernelSource() => @"
uint rng(uint s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
float rf(uint* s) { *s=rng(*s); return (*s&0xFFFFFF)/(float)0xFFFFFF; }
__kernel void monte_carlo_volume(__global uchar* out, int w, int h, int spr, float zoom) {
	int i = get_global_id(0); if(i>=w*h) return;
	int px=i%w, py=i/w;
	uint seed=(uint)(i*12345+1);
	float sumR=0, sumG=0, sumB=0;
	for(int s=0;s<spr;s++) {
		float rx = px + rf(&seed) - 0.5f;
		float ry = py + rf(&seed) - 0.5f;
		int xr = ((int)rx ^ (int)ry) & 255;
		float t = (float)(((int)rx ^ (int)ry) & 7) / 7.0f;
		sumR += xr * (0.7f + 0.3f * t);
		sumG += xr * (0.5f + 0.5f * (1-t));
		sumB += xr * (0.8f + 0.2f * t);
	}
	int oi=i*3;
	out[oi]=(uchar)(sumR/spr); out[oi+1]=(uchar)(sumG/spr); out[oi+2]=(uchar)(sumB/spr);
}";
}
