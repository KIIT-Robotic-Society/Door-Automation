#include <iostream>
#include <chrono>
#include <thread>
#include "VL53L0X.hpp"
#include "single.hpp"

//class implementation of VL53L0X ToF sensor with safe initialization, timeout handling, and error-resilient read operations.

DistanceSensor::DistanceSensor(uint32_t timing_budget_us, uint32_t timeout_ms)
    : timing_budget(timing_budget_us), timeout(timeout_ms) {}

// initialize the VL53L0X sensor
// - configures communication
// - applies timeout
// - sets measurement timing budget
// returns:
//   true  → sensor initialized successfully
//   false → hardware or communication failure

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


// read a single distance measurement in millimeters
// - returns a 16-bit distance value
// - in case of sensor failure or exception, return 8096 (custom error code)


uint16_t DistanceSensor::read() {
    try {
        return sensor.readRangeSingleMillimeters();
    } catch (...) {
        return 8096; // default error reading
    }
}

// check whether the sensor triggered a timeout on the last read

bool DistanceSensor::timeoutOccurred() {
    return sensor.timeoutOccurred();
}
