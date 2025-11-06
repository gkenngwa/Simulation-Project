using Godot;
using System;
using System.Threading;

[GlobalClass]
public partial class mandelbrot : RefCounted
{
	private const int maxIterations = 10000;
	private static int width, height;
	
	private static byte[] data;
	private static int threadCount;
	
	private static float offsetX = 0.75f;
	private static float zoomLevel = 4.0f;
	
	public static void ChangeTextureSize(int newWidth, int newHeight)
	{
		width = newWidth;
		height = newHeight;
		data = new byte[newWidth * newHeight * 3];
	}
	
	public static void ChangeThreadCount(int newThreadCount) {
		threadCount = newThreadCount;
	}
		
	public static byte[] GetData() {
		return data;
	}
	
	public static void MandelbrotTiled()
	{
		System.Array.Clear(data, 0, data.Length);
		
		int tileWidth = width / threadCount;
		
		for (int t = 0; t < threadCount; t++)
		{
			int startX = t * tileWidth;
			int endX = (t == threadCount - 1) ? width : startX + tileWidth;
			
			Thread thread = new Thread(() => RenderMandelbrot(startX, endX));
			thread.IsBackground = true;
			thread.Start();
		}
	}

	private static void RenderMandelbrot(int startX, int endX)
	{
		for (int px = startX; px < endX; px++)
		{
			for (int py = 0; py < height; py++)
			{
				ProcessPixel(px, py);
			}
		}
	}
	private static void ProcessPixel(int px, int py)
	{
		float x = (px - (offsetX * width)) / (width / zoomLevel);
		float y = (py - (width / zoomLevel)) / (width / zoomLevel);
		
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
		float hue = i / 255.0f;
		float saturation = 1.0f;
		float brightness = 0.75f;
		
		return Color.FromHsv(hue, saturation, brightness);
	}
}
