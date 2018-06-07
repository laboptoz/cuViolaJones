#pragma once
#ifndef ERRCHECK
#define CHECK(ans) { err_check((ans), __FILE__, __LINE__); }
inline void err_check(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#endif