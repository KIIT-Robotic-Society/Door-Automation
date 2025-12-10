#ifndef SINGLE_HPP
#define SINGLE_HPP

#include "VL53L0X.hpp"
#include <cstdint>
#include <cstdint>
#include <fstream>
#include <mutex>

// DistanceSensor
//   - Safe initialization
//   - Configurable timing budget & timeout
//   - Single-shot distance measurement API
//   - Timeout reporting for debugging / reliability checks

class DistanceSensor {
public:
    
    // Constructor
    //
    // Parameters:
    //   timing_budget_us : Measurement timing budget (accuracy vs speed tradeoff)
    //   timeout_ms        : Maximum allowed time for a measurement before timeout
    
    DistanceSensor(uint32_t timing_budget_us = 50000, uint32_t timeout_ms = 200);
        
    // begin()
    // -------
    // Initializes the VL53L0X sensor.
    // Returns:
    //   true  → initialization successful
    //   false → sensor or communication failure
    
    bool begin();
        
    // read()
    // ------
    // Performs a single distance measurement.
    // Returns:
    //   0–2000 mm : valid measured distance
    //   8096      : error code used when the reading fails
    
    uint16_t read();
        
    // timeoutOccurred()
    // -----------------
    // Indicates whether the last measurement triggered a timeout.
    
    bool timeoutOccurred();

private:
    VL53L0X sensor;        // Underlying hardware driver
    uint32_t timing_budget; // Measurement timing budget (µs)
    uint32_t timeout;       // Measurement timeout (ms)
};



#endif
