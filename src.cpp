// client_menu.cpp
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <curl/curl.h>
#include "json.hpp"

using json = nlohmann::json;
const std::string API_URL = "http://127.0.0.1:8000";
const std::string API_KEY = "uBJjvkPOIFJguPO"; // api key

std::atomic<bool> q_flag(false);
std::atomic<bool> live_started(false);
std::atomic<bool> heartbeat_running(true);
std::atomic<bool> live_polling_running(false);

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
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
            try { j = json::parse(readBuffer); } catch(...) { j = json::object(); }
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
            try { j = json::parse(readBuffer); } catch(...) { j = json::object(); }
        }
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    return j;
}

void heartbeat_loop() {
    while(heartbeat_running) {
        get_json(API_URL + "/heartbeat"); // ping silently
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void live_polling_loop() {
    live_polling_running = true;
    int consecutive_arka = 0;

    while(live_polling_running) {
        json status = get_json(API_URL + "/live/status");

        if(status.contains("name") && status["name"].is_string()) {
            std::string name = status["name"];
            if(name != "Unknown") {
                if(name == "Arka") {
                    consecutive_arka++;
                    if(consecutive_arka >= 2) { 
                        q_flag = true;
                        std::this_thread::sleep_for(std::chrono::seconds(3));
                        q_flag = false;
                        consecutive_arka = 0;
                        std::cout << "[LIVE STATUS] Detected: Arka [Q FLAG ACTIVATED]\n";
                    }
                } else {
                    consecutive_arka = 0;
                }

                if(name != "Arka") {
                    std::cout << "[LIVE STATUS] Detected: " << name << "\n";
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void list_faces() {
    json j = get_json(API_URL + "/faces");
    std::cout << j.dump(4) << "\n";
}

void add_face_camera() {
    std::string name;
    std::cout << "Enter name to add: ";
    std::cin >> name;
    std::string url = API_URL + "/face/add_camera?name=" + name;
    json j = post_json(url);
    std::cout << j.dump(4) << "\n";
}

void start_live() {
    if(live_started) {
        std::cout << "[INFO] Live already running\n";
        return;
    }
    json j = post_json(API_URL + "/live/start");
    if(j.contains("status") && (j["status"] == "started" || j["status"] == "already_running")) {
        live_started = true;
        std::cout << "[INFO] Live started\n";
        std::thread(live_polling_loop).detach();
    } else {
        std::cout << j.dump(4) << "\n";
    }
}

void stop_live() {
    if(!live_started) {
        std::cout << "[INFO] Live is not running\n";
        return;
    }
    post_json(API_URL + "/live/stop");
    live_polling_running = false;
    live_started = false;
    std::cout << "[INFO] Live stopped\n";
}

void get_logs() {
    json j = get_json(API_URL + "/logs");
    std::cout << j.dump(4) << "\n";
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::thread heartbeat_thread(heartbeat_loop);
    heartbeat_thread.detach();

    int choice;
    while(true) {
        std::cout << "\n--- MENU ---\n";
        std::cout << "1. List faces\n";
        std::cout << "2. Add face by camera\n";
        std::cout << "3. Start live\n";
        std::cout << "4. Stop live\n";
        std::cout << "5. Get logs\n";
        std::cout << "6. Exit\n";
        std::cout << "Choice: ";
        std::cin >> choice;

        switch(choice) {
            case 1: list_faces(); break;
            case 2: add_face_camera(); break;
            case 3: start_live(); break;
            case 4: stop_live(); break;
            case 5: get_logs(); break;
            case 6:
                heartbeat_running = false;
                live_polling_running = false;
                curl_global_cleanup();
                return 0;
            default: std::cerr << "Invalid choice\n";
        }

        std::cout << "[INFO] q_flag = " << q_flag << "\n";
    }

    curl_global_cleanup();
    return 0;
}
