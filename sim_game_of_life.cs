using Godot;
using System;
using System.Threading;
using Silk.NET.OpenCL;

[GlobalClass]
public partial class sim_game_of_life : sim, IParameterized
{
	private float zoomLevel = 1.0f;
	private new int gridSize = 4096;
	private int generationsCompleted = 0;
	private int liveCellCount = 0;
	
	private nint gpuSimKernel, gpuRenderKernel;
	private bool gpuKernelsInitialized = false;
	
	private int minX, maxX, minY, maxY;
	
	public override string GetName() => "Game of Life";
	public override int GetGenerationsCompleted() => generationsCompleted;
	public override int GetLiveCellCount() => liveCellCount;
	
	public void SetParameter(string name, object value)
	{
		if (name.ToLower() == "zoom") zoomLevel = Convert.ToSingle(value);
	}
	
	public object GetParameter(string name) => name.ToLower() == "zoom" ? zoomLevel : null;
	public string[] GetParameterNames() => new[] { "zoom" };
	
	private void InitSeed(byte[] grid)
	{
		int cx = gridSize / 2, cy = gridSize / 2;
		minX = cx - 5; maxX = cx + 5;
		minY = cy - 5; maxY = cy + 5;
		
		for (int dy = -5; dy <= 5; dy++)
			for (int dx = -5; dx <= 5; dx++)
				if ((dx + dy) % 2 == 0)
					grid[(cy + dy) * gridSize + (cx + dx)] = 1;
	}
	
	protected override void RenderCPU()
	{
		InitializeGrids(gridSize);
		InitSeed(gridA);
		generationsCompleted = 0;
		
		StartTimedSimulation(SimulateStripe);
	}
	
	private void SimulateStripe(int threadIndex, int totalThreads)
	{
		while (!timeUp)
		{
			int workMinY = Math.Max(1, minY - 1);
			int workMaxY = Math.Min(gridSize - 2, maxY + 1);
			int workMinX = Math.Max(1, minX - 1);
			int workMaxX = Math.Min(gridSize - 2, maxX + 1);
			
			int totalRows = workMaxY - workMinY + 1;
			int rowsPerThread = totalRows / totalThreads;
			int startRow = workMinY + threadIndex * rowsPerThread;
			int endRow = (threadIndex == totalThreads - 1) ? workMaxY + 1 : workMinY + (threadIndex + 1) * rowsPerThread;
			
			int localMinX = gridSize, localMaxX = 0, localMinY = gridSize, localMaxY = 0;
			
			for (int y = startRow; y < endRow; y++)
			{
				for (int x = workMinX; x <= workMaxX; x++)
				{
					int n = 0;
					for (int dy = -1; dy <= 1; dy++)
						for (int dx = -1; dx <= 1; dx++)
							if (dx != 0 || dy != 0)
								n += gridA[(y + dy) * gridSize + (x + dx)];
					
					int idx = y * gridSize + x;
					byte newVal = (gridA[idx] == 0 && n == 2) ? (byte)1 : (byte)0;
					gridB[idx] = newVal;
					
					if (newVal == 1)
					{
						if (x < localMinX) localMinX = x;
						if (x > localMaxX) localMaxX = x;
						if (y < localMinY) localMinY = y;
						if (y > localMaxY) localMaxY = y;
					}
				}
			}
			
			barrier.SignalAndWait();
			
			if (threadIndex == 0)
			{
				minX = gridSize; maxX = 0; minY = gridSize; maxY = 0;
			}
			
			barrier.SignalAndWait();
			
			lock (barrier)
			{
				if (localMinX < minX) minX = localMinX;
				if (localMaxX > maxX) maxX = localMaxX;
				if (localMinY < minY) minY = localMinY;
				if (localMaxY > maxY) maxY = localMaxY;
			}
			
			barrier.SignalAndWait();
			
			if (threadIndex == 0)
			{
				SwapGrids();
				generationsCompleted++;
			}
			
			barrier.SignalAndWait();
		}
		
		if (threadIndex == 0)
		{
			liveCellCount = 0;
			for (int i = 0; i < gridA.Length; i++)
				liveCellCount += gridA[i];
			
			RenderGrid();
		}
		else
		{
			ThreadCompleted();
		}
	}
	
	private void RenderGrid()
	{
		float viewSize = gridSize / zoomLevel;
		float startX = (gridSize - viewSize) * 0.5f;
		float startY = (gridSize - viewSize) * 0.5f;
		
		for (int py = 0; py < height; py++)
		{
			for (int px = 0; px < width; px++)
			{
				int gx = (int)(startX + px * viewSize / width);
				int gy = (int)(startY + py * viewSize / height);
				
				byte val = 0;
				if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize)
					val = gridA[gy * gridSize + gx];
				
				SetPixelColor(px, py, (byte)(val * 255), (byte)(val * 255), (byte)(val * 255));
			}
		}
		
		FinishRender();
	}
	
	protected override void RenderGPU()
	{
		if (!gpuKernelsInitialized)
		{
			InitializeGPU();
			InitializeGPUGrids(gridSize);
			
			unsafe
			{
				int err;
				nint prog = cl.CreateProgramWithSource(gpuContext, 1, new[] { GetKernelSource() }, null, &err);
				nint dev = gpuDevice;
				cl.BuildProgram(prog, 1, &dev, "", null, null);
				
				fixed (byte* s = System.Text.Encoding.ASCII.GetBytes("gol_step\0"))
				fixed (byte* r = System.Text.Encoding.ASCII.GetBytes("gol_render\0"))
				{
					gpuSimKernel = cl.CreateKernel(prog, s, &err);
					gpuRenderKernel = cl.CreateKernel(prog, r, &err);
				}
			}
			gpuKernelsInitialized = true;
		}
		
		unsafe
		{
			byte[] grid = new byte[gridSize * gridSize];
			InitSeed(grid);
			
			fixed (byte* p = grid)
				cl.EnqueueWriteBuffer(gpuQueue, gpuGridA, true, 0, (nuint)(gridSize * gridSize), p, 0, null, null);
			
			int gs = gridSize;
			nuint globalSize = (nuint)(gridSize * gridSize);
			nuint localSize = (nuint)workGroupCount;
			nint read = gpuGridA, write = gpuGridB;
			
			generationsCompleted = 0;
			var start = DateTime.Now;
			
			while ((DateTime.Now - start).TotalMilliseconds < 1000)
			{
				cl.SetKernelArg(gpuSimKernel, 0, (nuint)sizeof(nint), &read);
				cl.SetKernelArg(gpuSimKernel, 1, (nuint)sizeof(nint), &write);
				cl.SetKernelArg(gpuSimKernel, 2, (nuint)sizeof(int), &gs);
				cl.EnqueueNdrangeKernel(gpuQueue, gpuSimKernel, 1, null, &globalSize, &localSize, 0, null, null);
				cl.Finish(gpuQueue);
				(read, write) = (write, read);
				generationsCompleted++;
			}
			
			byte[] final = new byte[gridSize * gridSize];
			fixed (byte* p = final)
				cl.EnqueueReadBuffer(gpuQueue, read, true, 0, (nuint)(gridSize * gridSize), p, 0, null, null);
			
			liveCellCount = 0;
			for (int i = 0; i < final.Length; i++)
				liveCellCount += final[i];
			
			int w = width, h = height;
			float z = zoomLevel;
			nint buf = gpuBuffer;
			cl.SetKernelArg(gpuRenderKernel, 0, (nuint)sizeof(nint), &read);
			cl.SetKernelArg(gpuRenderKernel, 1, (nuint)sizeof(nint), &buf);
			cl.SetKernelArg(gpuRenderKernel, 2, (nuint)sizeof(int), &gs);
			cl.SetKernelArg(gpuRenderKernel, 3, (nuint)sizeof(int), &w);
			cl.SetKernelArg(gpuRenderKernel, 4, (nuint)sizeof(int), &h);
			cl.SetKernelArg(gpuRenderKernel, 5, (nuint)sizeof(float), &z);
			
			nuint renderSize = (nuint)(width * height);
			cl.EnqueueNdrangeKernel(gpuQueue, gpuRenderKernel, 1, null, &renderSize, &localSize, 0, null, null);
			cl.Finish(gpuQueue);
			
			fixed (byte* p = data)
				cl.EnqueueReadBuffer(gpuQueue, gpuBuffer, true, 0, (nuint)(width * height * 3), p, 0, null, null);
			
			FinishRender();
		}
	}
	
	protected override string GetKernelName() => "gol_step";
	protected override void SetKernelArgs() { }
	
	protected override string GetKernelSource() => @"
__kernel void gol_step(__global uchar* in, __global uchar* out, int gs) {
	int i = get_global_id(0);
	if (i >= gs * gs) return;
	int x = i % gs, y = i / gs;
	if (x == 0 || x == gs-1 || y == 0 || y == gs-1) { out[i] = 0; return; }
	int n = 0;
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
			if (dx || dy) n += in[(y+dy)*gs+(x+dx)];
	out[i] = (!in[i] && n==2) ? 1 : 0;
}
__kernel void gol_render(__global uchar* g, __global uchar* o, int gs, int w, int h, float z) {
	int i = get_global_id(0);
	if (i >= w*h) return;
	int px = i%w, py = i/w, oi = i*3;
	float vs = gs/z, stx = (gs-vs)*0.5f, sty = (gs-vs)*0.5f;
	int sx = (int)(stx + px/(float)w*vs), sy = (int)(sty + py/(float)h*vs);
	if (sx<0||sx>=gs||sy<0||sy>=gs) { o[oi]=20; o[oi+1]=20; o[oi+2]=30; return; }
	uchar a = g[sy*gs+sx];
	o[oi]=a*255; o[oi+1]=a*255; o[oi+2]=a*255;
}";
}
