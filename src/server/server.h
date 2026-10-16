// cpp-httplib server exposing the embedded UI + inference API.
#pragma once

#include "core/common.h"
#include "core/pipeline.h"

#include <string>
#include <memory>

namespace seedvr2 {

struct ServerConfig {
    std::string host = "127.0.0.1";
    int         port = 8188;
    bool        open_browser = true;
    // repository root; backend.json is written/read here (empty = CWD)
    std::string root;
};

class Server {
public:
    Server(Engine& engine, Progress& progress, ServerConfig cfg);
    void run();

private:
    Engine& engine_;
    Progress& progress_;
    ServerConfig cfg_;
};

} // namespace seedvr2
