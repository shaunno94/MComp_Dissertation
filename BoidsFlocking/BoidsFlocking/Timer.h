//Author: Shaun Heald - 120266942
//Provides an accurate timer for the game engine.
#pragma once
#include <Windows.h>
class Timer
{
public:
	Timer();
	void startTimer();
	void stopTimer();
	double split();
	double getLast();

private:

	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	double last = 1.0;
};