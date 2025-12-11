// compile: g++ src.cpp -o src -lcurl -pthread -lgpiodcxx

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <csignal>
#include <curl/curl.h>
#include "json.hpp"
#include <gpiod.hpp>
#include <single.hpp>
#include <fstream>
#include <ctime>
#include <iomanip>

using json = nlohmann::json;

// logging macros
#define INFO(msg)      log_message("INFO", msg)
#define WARN(msg)      log_message("WARNING", msg)
#define ERROR(msg)     log_message("ERROR", msg)
#define SENSORLOG(msg) log_message("SENSOR", msg)
#define CTRLLOG(msg)   log_message("CTRL", msg)

// configuration constants
const std::string API_URL = "http://127.0.0.1:8000";
const std::string API_KEY = "uBJjvkPOIFJguPO";
const std::string LOG_FILE = "system.log";

// sensor and timing configuration
constexpr int SENSOR_THRESHOLD_MM     = 200;  // distance threshold for triggering door
constexpr int LIVE_DURATION_SECONDS   = 30;   // max time to wait for face recognition
constexpr unsigned int GPIO_LINE      = 17;   // GPIO pin for door control
constexpr unsigned int GPIO_IDLE_LINE = 27;   // Indicator for idle state
constexpr unsigned int GPIO_LIVE_LINE = 22;   // Indicator for live/ML active
constexpr int COOLDOWN_SECONDS        = 60;   // prevent duplicate triggers for same person

// Polling and network configuration
constexpr int POLLING_INTERVAL_MS     = 200;  // how often to poll live status
constexpr int FACE_CONFIRM_COUNT      = 10;   // consecutive detections needed (10 * 200ms = 2 sec)
constexpr int HEARTBEAT_INTERVAL_SEC  = 5;    // API health check interval
constexpr int SENSOR_POLL_INTERVAL_MS = 50;   // sensor reading frequency
constexpr int CURL_TIMEOUT_SEC        = 10;   // HTTP request timeout
constexpr int THREAD_JOIN_TIMEOUT_SEC = 5;    // max wait time for thread shutdown

// GPIO timing
constexpr int GPIO_HIGH_DURATION_SEC  = 1;    // how long to keep door unlocked


// global state variables

// last recognized person tracking (for cooldown logic)
std::string last_triggered_name = "";
std::chrono::steady_clock::time_point last_trigger_time;
std::mutex last_trigger_mutex;  // protect access to above variables

// thread control flags
std::atomic<bool> program_running(true);
std::atomic<bool> heartbeat_running(true);
std::atomic<bool> live_polling_running(false);
std::atomic<bool> face_detected(false);
std::atomic<bool> live_semaphore(false);  // prevents sensor trigger while live is active
std::atomic<bool> api_ready(false);       // tracks API availability

// synchronization primitives
std::mutex mtx;
std::condition_variable cv;
bool sensor_trigger = false;

// thread handles
std::thread heartbeat_thread;
std::thread sensor_thread;
std::thread live_thread;
std::thread live_controller_thread;

// GPIO resources
std::unique_ptr<gpiod::chip> chip_ptr;
std::unique_ptr<gpiod::line> gpio_line_ptr;
std::unique_ptr<gpiod::line> gpio_idle_ptr;
std::unique_ptr<gpiod::line> gpio_live_ptr;


// logging system
std::mutex log_mutex;  // thread-safe logging

void log_message(const std::string& level, const std::string& msg) {
    std::lock_guard<std::mutex> lock(log_mutex);
    
    // get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << ms.count()
       << " [" << level << "] " << msg;
    
    std::string log_line = ss.str();
    
    // console output
    std::cout << log_line << std::endl;
    
    // file output
    std::ofstream log_file(LOG_FILE, std::ios::app);
    if (log_file.is_open()) {
        log_file << log_line << std::endl;
    }
}


// signal handler
void signal_handler(int) {
    INFO("SIGINT received. Shutting down...");
    program_running = false;
    heartbeat_running = false;
    cv.notify_all();
}


// HTTP UTILITIES WITH ERROR HANDLING

// RAII wrapper for CURL handle
class CurlHandle {
private:
    CURL* curl;
    
public:
    CurlHandle() : curl(curl_easy_init()) {
        if (curl) {
            // set timeout to prevent hanging
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT_SEC);
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CURL_TIMEOUT_SEC);
        }
    }
    
    ~CurlHandle() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }
    
    CURL* get() { return curl; }
    operator bool() const { return curl != nullptr; }
    
    // disable copy
    CurlHandle(const CurlHandle&) = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
};

// callback for writing HTTP response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// perform HTTP GET request with error handling
json get_json(const std::string& url) {
    CurlHandle curl;
    if (!curl) {
        ERROR("Failed to initialize CURL handle");
        return {};
    }

    std::string readBuffer;
    json j;

    // set up headers with API key
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("x-api-key: " + API_KEY).c_str());

    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl.get());
    
    if (res == CURLE_OK) {
        try { 
            j = json::parse(readBuffer); 
        } catch (const std::exception& e) {
            ERROR("JSON parse error: " + std::string(e.what()));
        }
    } else {
        ERROR("HTTP GET failed: " + std::string(curl_easy_strerror(res)));
    }
    
    curl_slist_free_all(headers);
    return j;
}

// perform HTTP POST request with error handling
json post_json(const std::string& url) {
    CurlHandle curl;
    if (!curl) {
        ERROR("Failed to initialize CURL handle");
        return {};
    }
    
    std::string readBuffer;
    json j;

    // set up headers with API key
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("x-api-key: " + API_KEY).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl.get(), CURLOPT_POST, 1);
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE, 0);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &readBuffer);

    CURLcode res = curl_easy_perform(curl.get());

    if (res == CURLE_OK) {
        try { 
            j = json::parse(readBuffer); 
        } catch (const std::exception& e) {
            ERROR("JSON parse error: " + std::string(e.what()));
        }
    } else {
        ERROR("HTTP POST failed: " + std::string(curl_easy_strerror(res)));
    }

    curl_slist_free_all(headers);
    return j;
}

bool gpio_init(unsigned int line = GPIO_LINE) {
    try {
        chip_ptr = std::make_unique<gpiod::chip>("gpiochip0");
    } catch (...) {
        ERROR("GPIOchip open FAILED");
        return false;
    }

    // door control
    try {
        gpiod::line raw = chip_ptr->get_line(line);
        gpio_line_ptr = std::make_unique<gpiod::line>(std::move(raw));

        gpiod::line_request config{
            "door",
            gpiod::line_request::DIRECTION_OUTPUT,
            0
        };
        gpio_line_ptr->request(config);
        gpio_line_ptr->set_value(0);

        INFO("Door GPIO initialized on line " + std::to_string(line));
    } catch (...) {
        ERROR("Door GPIO initialization FAILED (line " + std::to_string(line) + ")");
        return false; 
    }

    // idle indicator 
    try {
        gpiod::line idle_raw = chip_ptr->get_line(GPIO_IDLE_LINE);
        gpio_idle_ptr = std::make_unique<gpiod::line>(std::move(idle_raw));

        gpiod::line_request config{
            "idle-led",
            gpiod::line_request::DIRECTION_OUTPUT,
            0
        };
        gpio_idle_ptr->request(config);
        gpio_idle_ptr->set_value(1); // ON at boot

        INFO("Idle GPIO initialized on line " + std::to_string(GPIO_IDLE_LINE));
    } catch (...) {
        WARN("Idle GPIO (line " + std::to_string(GPIO_IDLE_LINE) +
             ") init failed, continuing without idle LED");
        gpio_idle_ptr.reset();
    }

    // live indicator 
    try {
        gpiod::line live_raw = chip_ptr->get_line(GPIO_LIVE_LINE);
        gpio_live_ptr = std::make_unique<gpiod::line>(std::move(live_raw));

        gpiod::line_request config{
            "live-led",
            gpiod::line_request::DIRECTION_OUTPUT,
            0
        };
        gpio_live_ptr->request(config);
        gpio_live_ptr->set_value(0); 

        INFO("Live GPIO initialized on line " + std::to_string(GPIO_LIVE_LINE));
    } catch (...) {
        WARN("Live GPIO (line " + std::to_string(GPIO_LIVE_LINE) +
             ") init failed, continuing without live LED");
        gpio_live_ptr.reset();
    }

    INFO("GPIO initialized: door=" + std::to_string(line) +
         ", idle=" + std::to_string(GPIO_IDLE_LINE) +
         ", live=" + std::to_string(GPIO_LIVE_LINE));

    return true;
}


void gpio_set(int v) {
    if (!gpio_line_ptr) return;
    gpio_line_ptr->set_value(v);
}

void gpio_idle_set(int v) {
    if (!gpio_idle_ptr) return;
    try {
        gpio_idle_ptr->set_value(v);
    } catch (...) {
        ERROR("Idle GPIO set failed");
    }
}

void gpio_live_set(int v) {
    if (!gpio_live_ptr) return;
    try {
        gpio_live_ptr->set_value(v);
    } catch (...) {
        ERROR("Live GPIO set failed");
    }
}


void gpio_cleanup() {
    try {
        if (gpio_line_ptr) {
            gpio_line_ptr->set_value(0);
            gpio_line_ptr->release();
            gpio_line_ptr.reset();
        }
        if (gpio_idle_ptr) {
            gpio_idle_ptr->set_value(0);
            gpio_idle_ptr->release();
            gpio_idle_ptr.reset();
        }
        if (gpio_live_ptr) {
            gpio_live_ptr->set_value(0);
            gpio_live_ptr->release();
            gpio_live_ptr.reset();
        }
        chip_ptr.reset();
        INFO("GPIO cleaned up successfully");
    } catch (const std::exception& e) {
        ERROR("GPIO cleanup error: " + std::string(e.what()));
    }
}


// Heartbeat thread - monitor API health
void heartbeat_loop() {
    INFO("Heartbeat thread started");
    
    while (heartbeat_running && program_running) {
        json response = get_json(API_URL + "/heartbeat");
        
        // Check if API is responding correctly
        if (response.empty() || !response.contains("status")) {
            WARN("Heartbeat failed - API may be down");
            api_ready = false;
        } else if (response["status"] == "live") {
            // API is up and running
            if (!api_ready) {
                INFO("API connection established");
            }
            api_ready = true;
        } else {
            WARN("Unexpected heartbeat status: " + response["status"].get<std::string>());
            api_ready = false;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(HEARTBEAT_INTERVAL_SEC));
    }
    
    INFO("Heartbeat thread stopped");
}

// Live polling thread - check for face recognition results
void live_polling_loop() {
    INFO("Live polling thread started");
    
    int count = 0;  // consecutive detection counter
    std::string currentName = "";

    while (live_polling_running && program_running) {

        // check cooldown status (thread-safe)
        {
            std::lock_guard<std::mutex> lock(last_trigger_mutex);
            if (!last_triggered_name.empty()) {
                auto now  = std::chrono::steady_clock::now();
                auto diff = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_trigger_time).count();

                if (diff >= COOLDOWN_SECONDS) {
                    INFO("Cooldown expired → resetting last_triggered_name");
                    last_triggered_name.clear();
                }
            }
        }

        // poll API for current detection status
        json st = get_json(API_URL + "/live/status");

        if (st.contains("name")) {
            std::string n = st["name"];

            // skip if this person is in cooldown period
            {
                std::lock_guard<std::mutex> lock(last_trigger_mutex);
                if (!last_triggered_name.empty() && n == last_triggered_name) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
                    continue;
                }
            }

            // only process valid recognitions (not Unknown or Spoof)
            if (n != "Unknown" && n != "Spoof") {

                // check for consistent detection
                if (n == currentName) {
                    count++;
                } else {
                    currentName = n;
                    count = 1;
                }

                // require multiple consecutive detections to avoid false triggers
                if (count >= FACE_CONFIRM_COUNT) {
                    INFO("Face confirmed: " + n + " (" + std::to_string(count) + " detections)");
                    
                    // activate door unlock
                    INFO("GPIO-HIGH for " + n);
                    gpio_set(1);
                    if (gpio_live_ptr) gpio_live_ptr->set_value(0);  // ensure live indicator off during unlock
                    std::this_thread::sleep_for(std::chrono::seconds(GPIO_HIGH_DURATION_SEC));
                    INFO("GPIO-LOW");
                    gpio_set(0);
                    if (gpio_idle_ptr) gpio_idle_ptr->set_value(1);   // return to idle mode


                    // update state (thread-safe)
                    face_detected = true;
                    {
                        std::lock_guard<std::mutex> lock(last_trigger_mutex);
                        last_triggered_name = n;
                        last_trigger_time   = std::chrono::steady_clock::now();
                    }

                    break;  // exit polling loop - face recognized
                }
            } else {
                // reset counter if detection is invalid
                if (count > 0) {
                    count = 0;
                    currentName = "";
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
    }
    
    INFO("Live polling thread stopped");
}

void start_live() {
    INFO("Starting live recognition...");

    // Set LED states
    if (gpio_idle_ptr) gpio_idle_ptr->set_value(0);   // idle OFF
    if (gpio_live_ptr) gpio_live_ptr->set_value(1);   // live ON
    
    json j = post_json(API_URL + "/live/start");

    // verify start was successful
    if (!(j.contains("status") &&
          (j["status"] == "started" || j["status"] == "already_running"))) {

        ERROR("Live start FAILED: " + j.dump());
        
        // restore LED states on failure
        if (gpio_live_ptr) gpio_live_ptr->set_value(0);
        if (gpio_idle_ptr) gpio_idle_ptr->set_value(1);
        return;
    }

    INFO("Live process launched — waiting for ML worker to initialize...");

    // ------------ FIXED: wait for Python ML to be ready ------------
    bool ml_ready = false;
    for (int i = 0; i < 100; i++) {  // wait up to 10 seconds (increased timeout)
        json st = get_json(API_URL + "/live/status");

        if (!st.contains("status")) {
            // No status yet, keep waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::string status = st["status"].get<std::string>();
        
        // Log status for debugging
        if (i % 10 == 0) {  // Log every second
            INFO("Current ML status: " + status);
        }

        // CRITICAL FIX: Accept EITHER "ready" OR "running" as success
        // The status might change from ready→running very quickly
        if (status == "ready" || status == "running") {
            INFO("ML worker is initialized (status: " + status + ")");
            ml_ready = true;
            break;
        }
        
        // Check for error state
        if (status == "error") {
            ERROR("ML worker reported error state");
            if (st.contains("error")) {
                ERROR("Error details: " + st["error"].get<std::string>());
            }
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!ml_ready) {
        ERROR("ML worker did not initialize in time. Aborting live mode.");
        
        // Get final status for debugging
        json final_st = get_json(API_URL + "/live/status");
        ERROR("Final status: " + final_st.dump());
        
        // fail closed — revert LED states
        if (gpio_live_ptr) gpio_live_ptr->set_value(0);
        if (gpio_idle_ptr) gpio_idle_ptr->set_value(1);
        return;
    }
    // ---------------------------------------------------------------

    INFO("Live recognition fully initialized — starting polling loop.");

    // reset state and start polling
    face_detected = false;
    live_polling_running = true;
    live_semaphore = true;

    live_thread = std::thread(live_polling_loop);
}


void stop_live() {
    INFO("Stopping live recognition...");
    
    // signal polling thread to stop
    live_polling_running = false;
    
    // wait for thread with timeout
    if (live_thread.joinable()) {
        auto start = std::chrono::steady_clock::now();
        while (live_thread.joinable()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
                
            if (elapsed >= THREAD_JOIN_TIMEOUT_SEC) {
                WARN("Live thread did not stop in time, detaching");
                live_thread.detach();
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            if (!live_polling_running && live_thread.joinable()) {
                live_thread.join();
                break;
            }
        }
    }
    
    // stop live feed on API side
    post_json(API_URL + "/live/stop");

    INFO("Live recognition stopped");

    // restore LED states
    if (gpio_live_ptr) gpio_live_ptr->set_value(0);   // live OFF
    if (gpio_idle_ptr) gpio_idle_ptr->set_value(1);   // idle ON
    live_semaphore = false;
}


// sensor thread - Monitor distance sensor for triggers
void sensor_loop() {
    INFO("Sensor thread started");

    DistanceSensor ds(50000, 200);
    
    try {
        ds.begin();
    } catch (const std::exception& e) {
        ERROR("Sensor initialization failed: " + std::string(e.what()));
        return;
    }

    while (program_running) {
        uint16_t d = ds.read();

        if (!ds.timeoutOccurred()) {
            // trigger only if live is not active and distance below threshold
            if (!live_semaphore && d < SENSOR_THRESHOLD_MM) {

                SENSORLOG("Trigger @ " + std::to_string(d) + " mm");

                // signal the live controller thread
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    sensor_trigger = true;
                }

                cv.notify_one();
                
                // debounce delay to prevent multiple rapid triggers
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        } else {
            WARN("Sensor timeout occurred");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(SENSOR_POLL_INTERVAL_MS));
    }
    
    INFO("Sensor thread stopped");
}

// live controller thread - sensor triggers and live recognition
void live_controller_loop() {
    INFO("Live controller thread started");
    
    while (program_running) {

        // wait for sensor trigger
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [] { return sensor_trigger || !program_running; });

        if (!program_running) break;

        SENSORLOG("Sensor trigger received");
        sensor_trigger = false;
        lk.unlock();

        // wait until API is ready (checked by heartbeat thread)
        int retry_count = 0;
        const int MAX_RETRIES = 30; // 30 seconds max wait
        
        while (program_running && !api_ready) {
            if (retry_count == 0) {
                WARN("Waiting for API to be ready...");
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
            retry_count++;
            
            if (retry_count >= MAX_RETRIES) {
                ERROR("API not ready after 30 seconds, skipping this trigger");
                break;
            }
        }

        // skip if program is shutting down or API timeout
        if (!program_running || retry_count >= MAX_RETRIES) {
            continue;
        }

        INFO("API ready → Starting live recognition");

        // start live recognition
        start_live();
        auto start = std::chrono::steady_clock::now();

        // wait for face detection or timeout
        while (program_running) {

            if (face_detected) {
                CTRLLOG("Face detected → stopping LIVE");
                break;
            }

            if (std::chrono::steady_clock::now() - start >=
                std::chrono::seconds(LIVE_DURATION_SECONDS)) {

                CTRLLOG(std::to_string(LIVE_DURATION_SECONDS) + "-second timeout reached");
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
        }
        
        // stop live recognition
        stop_live();
    }
    
    INFO("Live controller thread stopped");
}


int main() {
    INFO("=== KRS Door Automation System Starting ===");
    
    // register signal handler for graceful shutdown
    signal(SIGINT, signal_handler);

    // initialize libraries
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    if (!gpio_init(GPIO_LINE)) {
        ERROR("GPIO initialization failed - exiting");
        return 1;
    }

    INFO("System initialized successfully");

    // start all threads
    heartbeat_thread       = std::thread(heartbeat_loop);
    sensor_thread          = std::thread(sensor_loop);
    live_controller_thread = std::thread(live_controller_loop);

    INFO("Waiting for API connection...");
    
    // wait for initial API connection before proceeding
    int wait_count = 0;
    while (!api_ready && program_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        wait_count++;
        
        if (wait_count % 5 == 0) {
            WARN("Still waiting for API... (" + std::to_string(wait_count) + "s)");
        }
    }
    
    if (api_ready) {
        INFO("API connected - System ready");
    }

    // main loop
    while (program_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    INFO("Initiating shutdown sequence...");

    heartbeat_running = false;
    cv.notify_all();

    if (heartbeat_thread.joinable())       heartbeat_thread.join();
    if (sensor_thread.joinable())          sensor_thread.join();
    if (live_controller_thread.joinable()) live_controller_thread.join();
    if (live_thread.joinable())            live_thread.join();

    // Cleanup resources
    gpio_cleanup();
    curl_global_cleanup();

    INFO("=== System shut down ===");
    return 0;
}