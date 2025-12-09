#ifndef SINGLE_HPP
#define SINGLE_HPP

#include "VL53L0X.hpp"
#include <cstdint>
#include <cstdint>
#include <fstream>
#include <mutex>

class DistanceSensor {
public:
    DistanceSensor(uint32_t timing_budget_us = 50000, uint32_t timeout_ms = 200);
    bool begin();
    uint16_t read();
    bool timeoutOccurred();

private:
    VL53L0X sensor;
    uint32_t timing_budget;
    uint32_t timeout;
};



#endif
