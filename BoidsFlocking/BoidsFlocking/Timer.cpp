#include "Timer.h"
Timer::Timer()
{
	QueryPerformanceFrequency(&frequency);
}

void Timer::startTimer()
{
	QueryPerformanceCounter(&start);
}

void Timer::stopTimer()
{
	QueryPerformanceCounter(&end);
	last = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
}

double Timer::getLast()
{
	return last;
}

double Timer::split()
{
	LARGE_INTEGER tmp;
	QueryPerformanceCounter(&tmp);
	return (tmp.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
}