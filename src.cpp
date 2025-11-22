// compile: g++ src.cpp -o src -lcurl -pthread -lgpiodcxx

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <curl/curl.h>
#include "json.hpp"
#include <gpiod.hpp>

using json = nlohmann::json;

const std::string API_URL = "http://127.0.0.1:8000";
const std::string API_KEY = "uBJjvkPOIFJguPO";

std::atomic<bool> q_flag(false);
std::atomic<bool> live_polling_running(false);
std::atomic<bool> heartbeat_running(true);

gpiod::chip chip("gpiochip0");  
gpiod::line gpio = chip.get_line(17); 

std::thread live_thread;     
std::thread heartbeat_thread;


static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

void gpio_call(){
gpiod::line_request config{
    "face-detector",
    gpiod::line_request::DIRECTION_OUTPUT,
    0
};
gpio.request(config);

}

json get_json(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;
    json j;

    if(curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("x-api-key: " + API_KEY).c_str());

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        if(res == CURLE_OK) {
            try { j = json::parse(readBuffer); } catch(...) {}
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return j;
}

json post_json(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;
    json j;

    if(curl) {
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("x-api-key: " + API_KEY).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        if(res == CURLE_OK) {
            try { j = json::parse(readBuffer); } catch(...) {}
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return j;
}


void heartbeat_loop() {
    while(heartbeat_running) {
        get_json(API_URL + "/heartbeat");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}


void live_polling_loop() {
    int iterative_detection = 0;
    std::string casting_name = "";
    bool casting_bit = true;

    while (live_polling_running) {
        json status = get_json(API_URL + "/live/status");

        if (status.contains("name") && status["name"].is_string()) {
            std::string name = status["name"];

            if (name != "Unknown" && name != "Spoof") {

                if (name == casting_name) {
                    iterative_detection++;
                } else {
                    casting_name = name;
                    iterative_detection = 1;
                    casting_bit = true;
                }

                if (casting_bit && iterative_detection >= 3) {

                    std::cout << "[STATUS] Detected: " << name << std::endl;
                    gpio.set_value(1);
                    std::cout<<"GPIO HIGH"<<std::endl;
                    q_flag = true;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    q_flag = false;
                    gpio.set_value(0);
                    std::cout<<"GPIO LOW"<<std::endl;

                    casting_bit = false;  
                }
                
            } else {
                iterative_detection = 0;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}


void start_live() {
    if(live_polling_running) {
        std::cout << "[INFO] Live is already running.\n";
        return;
    }

    json j = post_json(API_URL + "/live/start");
    if(j.contains("status") && (j["status"] == "started" || j["status"] == "already_running")) {
        
        live_polling_running = true;
        live_thread = std::thread(live_polling_loop);

        std::cout << "[INFO] Live started.\n";
    } 
    else {
        std::cout << j.dump(4) << "\n";
    }
}

void stop_live() {
    if(!live_polling_running) {
        std::cout << "[INFO] Live is not running.\n";
        return;
    }

    post_json(API_URL + "/live/stop");
    live_polling_running = false;

    if(live_thread.joinable()) live_thread.join();

    std::cout << "[INFO] Live stopped.\n";
}

void list_faces() {
    json j = get_json(API_URL + "/faces");
    std::cout << j.dump(4) << std::endl;
}

void get_logs() {
    json j = get_json(API_URL + "/logs");
    std::cout << j.dump(4) << std::endl;
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    gpio_call();
    

    heartbeat_thread = std::thread(heartbeat_loop);

    int choice;
    while(true) {
        std::cout << "--- MENU ---"<<std::endl;
        std::cout << "1. List faces"<<std::endl;
        std::cout << "2. Start live"<<std::endl;
        std::cout << "3. Stop live"<<std::endl;
        std::cout << "4. Get logs"<<std::endl;
        std::cout << "5. Exit"<<std::endl;
        std::cout << "Choice: "<<std::endl;
        std::cin >> choice;

        switch(choice) {
            case 1: list_faces(); break;
            case 2: start_live(); break;
            case 3: stop_live(); break;
            case 4: get_logs(); break;
            case 5:
                heartbeat_running = false;
                if(heartbeat_thread.joinable()) heartbeat_thread.join();

                live_polling_running = false;
                if(live_thread.joinable()) live_thread.join();

                curl_global_cleanup();
                std::cout << "[INFO] Exiting cleanly.\n";
                return 0;

            default:
                std::cout << "Invalid choice.\n";
        }

        std::cout << "[INFO] q_flag = " << q_flag << "\n";
    }

    return 0;
}
