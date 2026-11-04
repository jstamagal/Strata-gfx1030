// Turn selected serial capture regions into graph diamonds. Capture still uses one
// stream; only the dependencies of named, independent producers are changed.
#pragma once
#include <initializer_list>
#include <utility>
#include <vector>

namespace q36tp_branch {
#if defined(__HIPCC__)
using Node = hipGraphNode_t;
using Graph = hipGraph_t;
inline Node tail(cudaStream_t stream) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    const Node* dependencies = nullptr;
    size_t count = 0;
    if (hipStreamGetCaptureInfo_v2(stream, &status, nullptr, nullptr, &dependencies, &count) != hipSuccess ||
        status != hipStreamCaptureStatusActive || count != 1) return nullptr;
    return dependencies[0];
}
inline hipError_t edges(Graph g, Node* from, Node* to, size_t* count) { return hipGraphGetEdges(g, from, to, count); }
inline hipError_t add(Graph g, const Node* from, const Node* to) { return hipGraphAddDependencies(g, from, to, 1); }
inline hipError_t remove(Graph g, const Node* from, const Node* to) { return hipGraphRemoveDependencies(g, from, to, 1); }
constexpr auto success = hipSuccess;
#else
using Node = cudaGraphNode_t;
using Graph = cudaGraph_t;
inline Node tail(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const Node* dependencies = nullptr;
    size_t count = 0;
    if (cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, nullptr, &dependencies, &count) != cudaSuccess ||
        status != cudaStreamCaptureStatusActive || count != 1) return nullptr;
    return dependencies[0];
}
inline cudaError_t edges(Graph g, Node* from, Node* to, size_t* count) { return cudaGraphGetEdges(g, from, to, count); }
inline cudaError_t add(Graph g, const Node* from, const Node* to) { return cudaGraphAddDependencies(g, from, to, 1); }
inline cudaError_t remove(Graph g, const Node* from, const Node* to) { return cudaGraphRemoveDependencies(g, from, to, 1); }
constexpr auto success = cudaSuccess;
#endif

// A group is P, end(A), end(B), ... in capture order. The successor of each
// end is the first node of the next producer; the successor of the last end is
// the consumer. Rewire P -> each producer and each producer -> consumer.
struct Group { std::vector<Node> marks; };
struct Staggered { Node before, qkv, z, conv; };
struct Stats { size_t groups = 0, arms = 0, skipped = 0; bool edges_available = true; };
struct Recorder {
    std::vector<Group> groups;
    std::vector<Staggered> staggered;
    void clear() { groups.clear(); staggered.clear(); }
    void record(std::initializer_list<Node> marks) {
        if (marks.size() < 3) return;
        Group group{std::vector<Node>(marks)};
        for (size_t i = 0; i < group.marks.size(); ++i)
            if (!group.marks[i] || (i && group.marks[i] == group.marks[i - 1])) return;
        groups.push_back(std::move(group));
    }
    void record_staggered(Node before, Node qkv, Node z, Node conv) {
        if (before && qkv && z && conv && before != qkv && qkv != z && z != conv)
            staggered.push_back({before, qkv, z, conv});
    }
    Stats apply(Graph graph) const {
        Stats result{};
        size_t count = 0;
        if (edges(graph, nullptr, nullptr, &count) != success) {
            result.edges_available = false;
            result.skipped = groups.size() + staggered.size();
            return result;
        }
        std::vector<Node> from(count), to(count);
        if (count && edges(graph, from.data(), to.data(), &count) != success) {
            result.edges_available = false;
            result.skipped = groups.size() + staggered.size();
            return result;
        }
        auto successor = [&](Node n) {
            Node next = nullptr;
            for (size_t i = 0; i < count; ++i) if (from[i] == n) {
                if (next) return (Node) nullptr; // not a linear capture edge
                next = to[i];
            }
            return next;
        };
        for (const Group& group : groups) {
            const std::vector<Node>& m = group.marks;
            const Node consumer = successor(m.back());
            bool valid = consumer != nullptr;
            std::vector<Node> starts(m.size());
            for (size_t i = 1; i < m.size(); ++i) {
                starts[i] = successor(m[i - 1]);
                valid &= starts[i] != nullptr && starts[i] != consumer && m[i] != consumer;
            }
            if (!valid) { ++result.skipped; continue; }
            size_t moved = 0;
            for (size_t i = 2; i < m.size(); ++i) {
                // Add the safe edges first. If an API rejects an edit, the old
                // serial dependency remains and that region still computes correctly.
                if (add(graph, &m[0], &starts[i]) != success ||
                    add(graph, &m[i - 1], &consumer) != success ||
                    remove(graph, &m[i - 1], &starts[i]) != success) break;
                ++moved;
            }
            if (moved) { ++result.groups; result.arms += moved; }
            if (moved != m.size() - 2) ++result.skipped;
        }
        // Capture order: QKV, Z, conv+AB, GDN. Conv needs QKV, while Z
        // needs only the common input. GDN must wait for both conv and Z.
        for (const Staggered& s : staggered) {
            const Node z_first = successor(s.qkv);
            const Node conv_first = successor(s.z);
            const Node gdn_first = successor(s.conv);
            if (!z_first || !conv_first || !gdn_first) { ++result.skipped; continue; }
            if (add(graph, &s.before, &z_first) != success ||
                add(graph, &s.qkv, &conv_first) != success ||
                add(graph, &s.z, &gdn_first) != success) { ++result.skipped; continue; }
            if (remove(graph, &s.qkv, &z_first) != success ||
                remove(graph, &s.z, &conv_first) != success) { ++result.skipped; continue; }
            ++result.groups;
            ++result.arms;
        }
        return result;
    }
};
} // namespace q36tp_branch
