
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

using json = nlohmann::json;

#define INFO(msg)      std::cout << "[INFO] " << msg << std::endl
#define WARN(msg)      std::cout << "[WARNING] " << msg << std::endl
#define ERROR(msg)     std::cout << "[ERROR] " << msg << std::endl
#define SENSORLOG(msg) std::cout << "[SENSOR] " << msg << std::endl
#define CTRLLOG(msg)   std::cout << "[CTRL] " << msg << std::endl

const std::string API_URL = "http://127.0.0.1:8000";
const std::string API_KEY = "uBJjvkPOIFJguPO";

constexpr int SENSOR_THRESHOLD_MM     = 200;
constexpr int LIVE_DURATION_SECONDS   = 30;
constexpr unsigned int GPIO_LINE      = 17;
constexpr int COOLDOWN_SECONDS        = 60;   

std::string last_triggered_name = "";
std::chrono::steady_clock::time_point last_trigger_time;


std::atomic<bool> program_running(true);
std::atomic<bool> heartbeat_running(true);
std::atomic<bool> live_polling_running(false);
std::atomic<bool> face_detected(false);
std::atomic<bool> live_semaphore(false);

std::mutex mtx;
std::condition_variable cv;
bool sensor_trigger = false;

std::thread heartbeat_thread;
std::thread sensor_thread;
std::thread live_thread;
std::thread live_controller_thread;

std::unique_ptr<gpiod::chip> chip_ptr;
std::unique_ptr<gpiod::line> gpio_line_ptr;


void signal_handler(int) {
    INFO("SIGINT received. Shutting down...");
    program_running = false;
    heartbeat_running = false;
    cv.notify_all();
}


static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}


json get_json(const std::string& url) {
    CURL* curl = curl_easy_init(); 
    if (!curl) return {};

    std::string readBuffer;
    json j;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("x-api-key: " + API_KEY).c_str());

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    if (curl_easy_perform(curl) == CURLE_OK) {
        try { j = json::parse(readBuffer); } catch (...) {}
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return j;
}

json post_json(const std::string& url) {
    CURL* curl = curl_easy_init(); 
    if (!curl) return {};
    
    std::string readBuffer;
    json j;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("x-api-key: " + API_KEY).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

    if (curl_easy_perform(curl) == CURLE_OK) {
        try { j = json::parse(readBuffer); } catch (...) {}
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return j;
}


bool gpio_init(unsigned int line = GPIO_LINE) {
    try {
        chip_ptr = std::make_unique<gpiod::chip>("gpiochip0");
        gpiod::line raw = chip_ptr->get_line(line);
        gpio_line_ptr = std::make_unique<gpiod::line>(std::move(raw));

        gpiod::line_request config{
            "live-system",
            gpiod::line_request::DIRECTION_OUTPUT,
            0
        };
        gpio_line_ptr->request(config);
        gpio_line_ptr->set_value(0);

        INFO("GPIO initialized on line " << line);
        return true;
    } catch (...) {
        ERROR("GPIO initialization FAILED");
        return false;
    }
}

void gpio_set(int v) {
    if (!gpio_line_ptr) return;
    gpio_line_ptr->set_value(v);
}

void heartbeat_loop() {
    while (heartbeat_running && program_running) {
        get_json(API_URL + "/heartbeat");
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

void live_polling_loop() {
    int count = 0;
    std::string currentName = "";

    while (live_polling_running && program_running) {

        if (!last_triggered_name.empty()) {
            auto now  = std::chrono::steady_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - last_trigger_time).count();

            if (diff >= COOLDOWN_SECONDS) {
                INFO("Cooldown expired → resetting last_triggered_name");
                last_triggered_name.clear();
            }
        }

        json st = get_json(API_URL + "/live/status");

        if (st.contains("name")) {
            std::string n = st["name"];

            if (!last_triggered_name.empty() && n == last_triggered_name) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            if (n != "Unknown" && n != "Spoof") {

                if (n == currentName) {
                    count++;
                } else {
                    currentName = n;
                    count = 1;
                }

                if (count >= 10) {
                    INFO("GPIO-HIGH for " << n);
                    gpio_set(1);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    INFO("GPIO-LOW");
                    gpio_set(0);

                    face_detected = true;
                    last_triggered_name = n;
                    last_trigger_time   = std::chrono::steady_clock::now();

                    break;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void start_live() {
    json j = post_json(API_URL + "/live/start");

    if (!(j.contains("status") &&
          (j["status"] == "started" || j["status"] == "already_running"))) {

        ERROR("Live start FAILED: " << j.dump());
        return;
    }

    face_detected = false;
    live_polling_running = true;
    live_semaphore = true;

    live_thread = std::thread(live_polling_loop);
}

void stop_live() {
    live_polling_running = false;
    post_json(API_URL + "/live/stop");

    if (live_thread.joinable())
        live_thread.join();

    INFO("LIVE STOPPED");
    live_semaphore = false;
}

void sensor_loop() {
    INFO("Sensor thread started");

    DistanceSensor ds(50000, 200);
    ds.begin();

    while (program_running) {
        uint16_t d = ds.read();

        if (!ds.timeoutOccurred()) {
            if (!live_semaphore && d < SENSOR_THRESHOLD_MM) {

                SENSORLOG("Trigger @ " << d << " mm");

                {
                    std::lock_guard<std::mutex> lk(mtx);
                    sensor_trigger = true;
                }

                cv.notify_one();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void live_controller_loop() {
    while (program_running) {

        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [] { return sensor_trigger || !program_running; });

        if (!program_running) break;

        SENSORLOG("sensor trigger received");

        sensor_trigger = false;
        lk.unlock();

        start_live();

        auto start = std::chrono::steady_clock::now();

        while (program_running) {

            if (face_detected) {
                CTRLLOG("Face detected → stopping LIVE");
                break;
            }

            if (std::chrono::steady_clock::now() - start >=
                std::chrono::seconds(LIVE_DURATION_SECONDS)) {

                CTRLLOG("30-second timeout reached");
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        stop_live();
    }
}

int main() {
    signal(SIGINT, signal_handler);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    gpio_init(GPIO_LINE);

    INFO("System started");

    heartbeat_thread       = std::thread(heartbeat_loop);
    sensor_thread          = std::thread(sensor_loop);
    live_controller_thread = std::thread(live_controller_loop);

    while (program_running)
        std::this_thread::sleep_for(std::chrono::seconds(1));

    heartbeat_running = false;
    cv.notify_all();

    if (heartbeat_thread.joinable())       heartbeat_thread.join();
    if (sensor_thread.joinable())          sensor_thread.join();
    if (live_controller_thread.joinable()) live_controller_thread.join();
    if (live_thread.joinable())            live_thread.join();

    INFO("System shut down cleanly");
    curl_global_cleanup();
    return 0;
}
