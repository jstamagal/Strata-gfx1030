#pragma once
// CPU-only cache/index and routing-count I/O. No change to quantized payloads.
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace q36pack {
inline void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error("q36 pack/profile: " + why);
}
struct Entry { uint64_t offset, stride, up, down, bytes; int gu, dt; };
class Cache {
    std::ifstream data;
    std::mutex lock;
public:
    std::vector<Entry> layers;
    void open(const std::string& dir, const std::vector<std::string>& sources,
              size_t nl, size_t ne, size_t embd, size_t ff) {
        std::ifstream f(std::filesystem::path(dir) / "q36_experts.txt");
        std::string magic; size_t ver=0, l=0, e=0, n=0, h=0, ns=0;
        f >> magic >> ver >> l >> e >> n >> h >> ns;
        require(f && magic == "Q36PACK" && ver == 1 && l == nl && e == ne && n == embd && h == ff,
                "cache version or model geometry mismatch");
        require(ns == sources.size(), "source shard count mismatch");
        for (const auto& source : sources) {
            std::string name; uint64_t size=0; int64_t mtime=0;
            f >> std::quoted(name) >> size >> mtime;
#if defined(_WIN32)
            struct _stat64 st{};
            const int status = _stat64(source.c_str(), &st);
#else
            struct stat st{};
            const int status = stat(source.c_str(), &st);
#endif
            require(f && status == 0 && name == std::filesystem::path(source).filename().string() &&
                    size == uint64_t(st.st_size) && mtime == int64_t(st.st_mtime), "source identity mismatch: " + source);
        }
        const auto path = std::filesystem::path(dir) / "experts.bin";
        const uint64_t total = std::filesystem::file_size(path);
        layers.resize(nl);
        uint64_t end=0;
        for (size_t i=0; i<nl; ++i) {
            size_t id=nl; std::string hash; auto& x=layers[i];
            f >> id >> x.gu >> x.dt >> x.offset >> x.stride >> x.up >> x.down >> x.bytes >> hash;
            require(f && id == i && hash.size() == 64 && x.stride && x.bytes <= x.stride &&
                    x.offset % 4096 == 0 && x.offset >= end && x.offset <= total &&
                    ne <= (total-x.offset)/x.stride, "invalid/truncated layer index");
            end = x.offset + ne*x.stride;
            require(total-end >= 4096, "missing MMQ tail");
            end += 4096;
        }
        require(end == total, "cache size mismatch");
        f >> std::ws;
        require(f.eof(), "unexpected index records");
        data.open(path, std::ios::binary);
        require(bool(data), "cannot open experts.bin");
    }
    void read(size_t li, size_t first, size_t count, void* dst) {
        const auto& x=layers.at(li);
        std::lock_guard<std::mutex> guard(lock);
        data.seekg(x.offset + first*x.stride);
        data.read(static_cast<char*>(dst), count*x.stride);
        require(bool(data), "short cache read");
    }
};
// STRP v1: exactly the main engine's profile format. Require a complete, unique ranking.
inline std::vector<std::vector<int>> profile(const std::string& path, size_t nl, size_t ne) {
    std::ifstream f(path, std::ios::binary);
    char magic[4]{}; uint32_t h[5]{};
    f.read(magic,4); f.read(reinterpret_cast<char*>(h),sizeof h);
    const size_t disk_nl = h[1];
    require(f && !std::memcmp(magic,"STRP",4) && h[0]==1 &&
            (disk_nl==nl || disk_nl+1==nl) && h[2]==ne &&
            h[3]==disk_nl*ne && h[4]==disk_nl*ne, "profile must rank every main-layer pair for this geometry");
    std::vector<std::vector<int>> out(nl);
    std::vector<int32_t> table(disk_nl*ne, -1), disk(disk_nl*ne);
    for (size_t i=0; i<disk_nl*ne; ++i) {
        uint16_t p[2]{}; f.read(reinterpret_cast<char*>(p),sizeof p);
        require(f && p[0]<disk_nl && p[1]<ne, "bad profile pair");
        auto& rank=table[p[0]*ne+p[1]];
        require(rank<0,"duplicate profile pair"); rank=int32_t(i);
        out[p[0]].push_back(p[1]);
    }
    f.read(reinterpret_cast<char*>(disk.data()),disk.size()*sizeof(int32_t));
    require(f && disk==table && f.peek()==EOF, "truncated/inconsistent profile table");
    if (disk_nl < nl) {
        out.back().resize(ne);
        std::iota(out.back().begin(), out.back().end(), 0);
    }
    return out;
}
class Trace {
    std::string path;
    size_t nl=0, ne=0;
    std::vector<uint64_t> counts;
public:
    bool active() const { return !path.empty(); }
    void open(const std::string& p, size_t l, size_t e) {
        path=p; nl=l; ne=e; counts.assign(l*e,0); save();
    }
    void add(size_t l, const int32_t* ids, size_t n) {
        require(l<nl, "trace layer out of range");
        for(size_t i=0;i<n;++i) {
            require(ids[i]>=0 && size_t(ids[i])<ne,"trace expert out of range");
            ++counts[l*ne+ids[i]];
        }
    }
    void save() const {
        if (!active()) return;
        const std::string tmp=path+".tmp";
        std::ofstream f(tmp,std::ios::binary|std::ios::trunc);
        const uint32_t h[]={1,uint32_t(nl),uint32_t(ne)};
        f.write("Q36C",4); f.write(reinterpret_cast<const char*>(h),sizeof h);
        f.write(reinterpret_cast<const char*>(counts.data()),counts.size()*sizeof(uint64_t));
        f.close(); require(bool(f),"cannot write trace " + path);
#if defined(_WIN32)
        require(MoveFileExA(tmp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)!=0,
                "cannot publish trace " + path);
#else
        require(std::rename(tmp.c_str(),path.c_str())==0,"cannot publish trace " + path);
#endif
    }
};
} // namespace q36pack
