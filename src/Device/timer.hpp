// timer.hpp
#pragma once
#include <chrono>
#include <cstdio>

class ScopedTimer
{
public:
    explicit ScopedTimer(const char* name)
        : name_(name),
          start_(clock::now()) {}

    ~ScopedTimer()
    {
        auto end = clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::printf("[TIMER] %s : %.6f ms\n", name_, ms);
    }

private:
    using clock = std::chrono::steady_clock;
    const char* name_;
    clock::time_point start_;
};
