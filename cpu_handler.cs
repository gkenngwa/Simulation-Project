using Godot;
using System;
using System.Threading;

[GlobalClass]
public partial class cpu_handler : RefCounted
{
	private int threadCount;
	private int threadsCompleted;
	private Action onAllComplete;
	private Barrier barrier;
	private bool timeUp;
	
	public int ThreadCount => threadCount;
	public Barrier Barrier => barrier;
	public bool TimeUp => timeUp;
	
	public void RunParallel(int threads, Action<int, int> work, Action onComplete)
	{
		threadCount = threads;
		threadsCompleted = 0;
		onAllComplete = onComplete;
		
		for (int t = 0; t < threadCount; t++)
		{
			int idx = t;
			Thread thread = new Thread(() =>
			{
				work(idx, threadCount);
				OnThreadComplete();
			});
			thread.IsBackground = false;
			thread.Start();
		}
	}
	
	public void RunTimedSimulation(int threads, Action<int, int, cpu_handler> work, Action onComplete)
	{
		threadCount = threads;
		threadsCompleted = 0;
		onAllComplete = onComplete;
		timeUp = false;
		barrier = new Barrier(threadCount);
		
		new Thread(() =>
		{
			Thread.Sleep(1000);
			timeUp = true;
		}) { IsBackground = true }.Start();
		
		for (int t = 0; t < threadCount; t++)
		{
			int idx = t;
			new Thread(() =>
			{
				work(idx, threadCount, this);
				OnThreadComplete();
			}) { IsBackground = false }.Start();
		}
	}
	
	public void SignalAndWait()
	{
		barrier.SignalAndWait();
	}
	
	public void LockAndUpdate(Action action)
	{
		lock (barrier)
		{
			action();
		}
	}
	
	private void OnThreadComplete()
	{
		int completed = Interlocked.Increment(ref threadsCompleted);
		if (completed == threadCount)
		{
			onAllComplete?.Invoke();
		}
	}
}
