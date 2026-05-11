// Minimal safetensors reader. Header-only.
//
// File layout (https://huggingface.co/docs/safetensors):
//   [8 bytes LE u64: header_len]
//   [header_len bytes: JSON]
//   [remaining: raw tensor data, native endianness]
//
// JSON header form:
//   { "<name>": { "dtype": "F32", "shape": [...], "data_offsets": [a, b] }, ...,
//     "__metadata__": { ... }  // optional
//   }
// data_offsets are relative to the start of the data region (after the header).
//
// This loader only handles F32 tensors — that's all our models use. The JSON
// parser is a small hand-rolled subset that covers exactly the safetensors
// schema (strings, integer arrays, nested objects); not a general-purpose
// parser.

#pragma once

#include "cg.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace safetensors {

struct TensorEntry {
    std::string      dtype;          // "F32" for everything we touch
    std::vector<int> shape;
    size_t           offset_bytes;   // within the data region
    size_t           length_bytes;
};

struct File {
    std::unordered_map<std::string, TensorEntry> index;
    std::vector<char>                            data;   // raw bytes (the data region only)
};

// ---- Tiny JSON parser, just enough for the safetensors header schema ------

class JsonParser {
public:
    JsonParser(const char* s, size_t n) : p_(s), end_(s + n) {}

    void skip_ws() { while (p_ < end_ && std::isspace((unsigned char)*p_)) ++p_; }

    void expect(char c) {
        skip_ws();
        if (p_ >= end_ || *p_ != c)
            throw std::runtime_error(std::string("safetensors: expected '") + c
                                     + "' at offset " + std::to_string(p_ - begin_));
        ++p_;
    }

    bool match(char c) {
        skip_ws();
        if (p_ < end_ && *p_ == c) { ++p_; return true; }
        return false;
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (p_ < end_ && *p_ != '"') {
            if (*p_ == '\\' && p_ + 1 < end_) {
                ++p_;
                char e = *p_;
                if      (e == 'n') s += '\n';
                else if (e == 't') s += '\t';
                else                s += e;       // \" \\ \/ etc.
            } else {
                s += *p_;
            }
            ++p_;
        }
        expect('"');
        return s;
    }

    long long parse_int() {
        skip_ws();
        const char* start = p_;
        if (p_ < end_ && *p_ == '-') ++p_;
        while (p_ < end_ && std::isdigit((unsigned char)*p_)) ++p_;
        if (start == p_) throw std::runtime_error("safetensors: expected integer");
        return std::stoll(std::string(start, p_ - start));
    }

    std::vector<long long> parse_int_array() {
        expect('[');
        std::vector<long long> v;
        skip_ws();
        if (!match(']')) {
            v.push_back(parse_int());
            while (match(',')) v.push_back(parse_int());
            expect(']');
        }
        return v;
    }

    // Skip whatever JSON value is at the cursor (string, number, object, array, true/false/null).
    void skip_value() {
        skip_ws();
        if (p_ >= end_) throw std::runtime_error("safetensors: unexpected EOF in value");
        char c = *p_;
        if (c == '"') { parse_string(); return; }
        if (c == '{' || c == '[') {
            char open = c, close = (c == '{' ? '}' : ']');
            int depth = 0;
            while (p_ < end_) {
                if (*p_ == '"') { parse_string(); continue; }
                if      (*p_ == open)  ++depth;
                else if (*p_ == close) { --depth; if (depth == 0) { ++p_; return; } }
                ++p_;
            }
            throw std::runtime_error("safetensors: unterminated structure");
        }
        // number / true / false / null: read until separator.
        while (p_ < end_ && *p_ != ',' && *p_ != '}' && *p_ != ']'
                         && !std::isspace((unsigned char)*p_)) ++p_;
    }

    void set_begin(const char* b) { begin_ = b; }

private:
    const char* p_;
    const char* end_;
    const char* begin_ = nullptr;   // for error messages only
};

// ---- File load ------------------------------------------------------------

inline File load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);

    uint64_t hdr_len = 0;
    f.read(reinterpret_cast<char*>(&hdr_len), 8);
    if (!f) throw std::runtime_error("safetensors: short read on header length");

    std::string hdr(hdr_len, '\0');
    f.read(hdr.data(), (std::streamsize)hdr_len);
    if (!f) throw std::runtime_error("safetensors: short read on header");

    // Total file size to determine data region size.
    f.seekg(0, std::ios::end);
    size_t total = (size_t)f.tellg();
    size_t data_off = 8 + (size_t)hdr_len;
    if (total < data_off) throw std::runtime_error("safetensors: file truncated");
    size_t data_len = total - data_off;

    File out;
    out.data.resize(data_len);
    f.seekg((std::streamoff)data_off, std::ios::beg);
    f.read(out.data.data(), (std::streamsize)data_len);
    if ((size_t)f.gcount() != data_len)
        throw std::runtime_error("safetensors: short read on data region");

    // Parse the JSON header.
    JsonParser p(hdr.data(), hdr.size());
    p.set_begin(hdr.data());
    p.expect('{');
    bool first = true;
    while (true) {
        p.skip_ws();
        if (!first) {
            if (!p.match(',')) break;
        }
        first = false;
        p.skip_ws();
        // empty object {}?
        if (p.match('}')) return out;

        std::string name = p.parse_string();
        p.expect(':');

        if (name == "__metadata__") {
            p.skip_value();
            continue;
        }

        p.expect('{');
        TensorEntry e;
        bool inner_first = true;
        while (true) {
            if (!inner_first) {
                if (!p.match(',')) break;
            }
            inner_first = false;
            p.skip_ws();
            if (p.match('}')) { goto stored; }   // empty inner object — unusual
            std::string k = p.parse_string();
            p.expect(':');
            if      (k == "dtype")        e.dtype = p.parse_string();
            else if (k == "shape") {
                auto sh = p.parse_int_array();
                e.shape.reserve(sh.size());
                for (long long d : sh) e.shape.push_back((int)d);
            }
            else if (k == "data_offsets") {
                auto off = p.parse_int_array();
                if (off.size() != 2) throw std::runtime_error("safetensors: bad data_offsets");
                e.offset_bytes = (size_t)off[0];
                e.length_bytes = (size_t)(off[1] - off[0]);
            }
            else p.skip_value();
        }
        p.expect('}');
stored:
        out.index.emplace(std::move(name), std::move(e));
    }
    // We may exit the loop because we hit the closing '}' of the outer object.
    // (We don't require it here — the index is built up regardless.)
    return out;
}

// ---- Tensor accessors ----------------------------------------------------

// Copy a named tensor's data into an existing cg::Tensor. Shape and dtype must match.
inline void copy_into(const File& f, const std::string& name, cg::Tensor& dst) {
    auto it = f.index.find(name);
    if (it == f.index.end()) throw std::runtime_error("safetensors: missing '" + name + "'");
    const auto& e = it->second;
    if (e.dtype != "F32")
        throw std::runtime_error("safetensors: '" + name + "' dtype is " + e.dtype + ", expected F32");
    if (e.shape != dst.shape()) {
        std::string s = "safetensors: shape mismatch for '" + name + "' — file [";
        for (size_t i = 0; i < e.shape.size(); ++i) { s += std::to_string(e.shape[i]); if (i + 1 < e.shape.size()) s += ","; }
        s += "] vs dst [";
        for (size_t i = 0; i < dst.shape().size(); ++i) { s += std::to_string(dst.shape()[i]); if (i + 1 < dst.shape().size()) s += ","; }
        s += "]";
        throw std::runtime_error(s);
    }
    if (e.length_bytes != (size_t)dst.numel() * sizeof(float))
        throw std::runtime_error("safetensors: byte size mismatch for '" + name + "'");
    std::memcpy(dst.data(), f.data.data() + e.offset_bytes, e.length_bytes);
}

// Read a tensor by name into a new cg::Tensor.
inline cg::Tensor read_f32(const File& f, const std::string& name) {
    auto it = f.index.find(name);
    if (it == f.index.end()) throw std::runtime_error("safetensors: missing '" + name + "'");
    const auto& e = it->second;
    if (e.dtype != "F32")
        throw std::runtime_error("safetensors: '" + name + "' dtype is " + e.dtype + ", expected F32");
    int numel = 1;
    for (int d : e.shape) numel *= d;
    if (e.length_bytes != (size_t)numel * sizeof(float))
        throw std::runtime_error("safetensors: byte size mismatch for '" + name + "'");
    cg::Tensor t(e.shape);
    std::memcpy(t.data(), f.data.data() + e.offset_bytes, e.length_bytes);
    return t;
}

} // namespace safetensors
