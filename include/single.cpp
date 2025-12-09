#include <iostream>
#include <chrono>
#include <thread>
#include "VL53L0X.hpp"
#include "single.hpp"


DistanceSensor::DistanceSensor(uint32_t timing_budget_us, uint32_t timeout_ms)
    : timing_budget(timing_budget_us), timeout(timeout_ms) {}

bool DistanceSensor::begin() {
    try {
        sensor.initialize();
        sensor.setTimeout(timeout);
        sensor.setMeasurementTimingBudget(timing_budget);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "[DistanceSensor] Init failed: " << e.what() << std::endl;
        return false;
    }
}

uint16_t DistanceSensor::read() {
    try {
        return sensor.readRangeSingleMillimeters();
    } catch (...) {
        return 8096; // default error reading
    }
}

bool DistanceSensor::timeoutOccurred() {
    return sensor.timeoutOccurred();
}
