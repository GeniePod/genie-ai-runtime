// tokenizer.cpp — BPE tokenizer with GGUF vocabulary loading
//
// Reads vocabulary directly from GGUF metadata.
// Simple greedy BPE encode (not optimal but functional).
// For production: link against sentencepiece or use HF tokenizers C API.

#include "jllm_engine.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <climits>

namespace jllm {

static std::string utf8_from_codepoint(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back((char)cp);
    } else if (cp <= 0x7FF) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
    return out;
}

static int utf8_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static const std::vector<std::string>& byte_encoder() {
    static const std::vector<std::string> enc = [] {
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        std::vector<std::string> out(256);
        for (size_t i = 0; i < bs.size(); ++i)
            out[bs[i]] = utf8_from_codepoint((uint32_t)cs[i]);
        return out;
    }();
    return enc;
}

static const std::unordered_map<std::string, unsigned char>& byte_decoder() {
    static const std::unordered_map<std::string, unsigned char> dec = [] {
        std::unordered_map<std::string, unsigned char> out;
        const auto& enc = byte_encoder();
        for (int i = 0; i < 256; ++i)
            out[enc[i]] = (unsigned char)i;
        return out;
    }();
    return dec;
}

static std::string encode_bytes_gpt2(const std::string& text) {
    const auto& enc = byte_encoder();
    std::string out;
    out.reserve(text.size() * 2);
    for (unsigned char c : text)
        out += enc[c];
    return out;
}

static std::vector<std::string> split_utf8_symbols(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size();) {
        int len = utf8_len((unsigned char)text[i]);
        if (i + len > text.size()) len = 1;
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

static std::vector<std::string> pretokenize_ascii(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;

        if (text[i] == '\'' && i + 1 < text.size()) {
            static const char* contractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (const char* c : contractions) {
                size_t n = strlen(c);
                if (i + 1 + n <= text.size() && text.compare(i + 1, n, c) == 0) {
                    out.push_back(text.substr(i, n + 1));
                    i += n + 1;
                    goto next_piece;
                }
            }
        }

        if (text[i] == ' ' && i + 1 < text.size() &&
            !std::isspace((unsigned char)text[i + 1])) {
            i++;
        }

        if (i < text.size() && std::isalpha((unsigned char)text[i])) {
            while (i < text.size() && std::isalpha((unsigned char)text[i])) i++;
        } else if (i < text.size() && std::isdigit((unsigned char)text[i])) {
            while (i < text.size() && std::isdigit((unsigned char)text[i])) i++;
        } else if (i < text.size() && !std::isspace((unsigned char)text[i])) {
            while (i < text.size() &&
                   !std::isspace((unsigned char)text[i]) &&
                   !std::isalpha((unsigned char)text[i]) &&
                   !std::isdigit((unsigned char)text[i])) i++;
        } else {
            while (i < text.size() && std::isspace((unsigned char)text[i])) i++;
        }

        out.push_back(text.substr(start, i - start));

next_piece:
        continue;
    }
    return out;
}

static std::string bpe_key(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + b.size() + 1);
    key += a;
    key.push_back('\n');
    key += b;
    return key;
}

bool Tokenizer::load_from_gguf(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Read GGUF header
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    if (magic != 0x46554747) { fclose(f); return false; }

    // Scan metadata for tokenizer entries
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t key_len;
        fread(&key_len, 8, 1, f);
        if (key_len > 1024) { fseek(f, key_len, SEEK_CUR); fseek(f, 4, SEEK_CUR); continue; }

        char key[1025] = {};
        fread(key, 1, key_len, f);

        uint32_t vtype;
        fread(&vtype, 4, 1, f);

        // tokenizer.ggml.tokens = array of strings
        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vtype == 9) { // GGUF_TYPE_ARRAY
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);

            vocab.resize(arr_len);
            for (uint64_t t = 0; t < arr_len; t++) {
                uint64_t str_len;
                fread(&str_len, 8, 1, f);
                vocab[t].resize(str_len);
                fread(&vocab[t][0], 1, str_len, f);
            }
            fprintf(stderr, "[tokenizer] Loaded %zu tokens from GGUF\n", vocab.size());
        }
        // tokenizer.ggml.merges = array of "left right" BPE merge strings
        else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vtype == 9) {
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);

            bpe_ranks_.clear();
            bpe_ranks_.reserve(arr_len);
            for (uint64_t m = 0; m < arr_len; m++) {
                uint64_t str_len;
                fread(&str_len, 8, 1, f);
                std::string merge(str_len, '\0');
                fread(&merge[0], 1, str_len, f);

                size_t sp = merge.find(' ');
                if (sp != std::string::npos) {
                    std::string left = merge.substr(0, sp);
                    std::string right = merge.substr(sp + 1);
                    bpe_ranks_[bpe_key(left, right)] = (int)m;
                }
            }
            fprintf(stderr, "[tokenizer] Loaded %zu BPE merges from GGUF\n", bpe_ranks_.size());
        }
        // tokenizer.ggml.scores = array of float32 unigram piece scores (SPM)
        else if (strcmp(key, "tokenizer.ggml.scores") == 0 && vtype == 9) {
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);
            token_scores_.resize(arr_len);
            if (arr_type == 6) {  // float32
                fread(token_scores_.data(), sizeof(float), arr_len, f);
            } else {
                int esz = (arr_type <= 1 || arr_type == 7) ? 1
                          : (arr_type <= 3) ? 2 : (arr_type <= 6) ? 4 : 8;
                fseek(f, (long)arr_len * esz, SEEK_CUR);
            }
            fprintf(stderr, "[tokenizer] Loaded %zu unigram scores from GGUF\n",
                    token_scores_.size());
        }
        else if (strcmp(key, "tokenizer.ggml.model") == 0 && vtype == 8) {
            uint64_t str_len;
            fread(&str_len, 8, 1, f);
            std::string model(str_len, '\0');
            fread(&model[0], 1, str_len, f);
            byte_encode_ = (model == "gpt2" || model == "qwen2");
            // "llama" is GGUF's label for the SentencePiece/unigram tokenizer
            // used by Gemma and Llama-SPM models.
            is_spm_ = (model == "llama");
        }
        else if (strcmp(key, "tokenizer.ggml.add_space_prefix") == 0 && vtype == 7) {
            uint8_t v = 0;
            fread(&v, 1, 1, f);
            add_space_prefix_ = (v != 0);
        }
        // tokenizer.ggml.bos_token_id
        else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && vtype == 4) {
            fread(&bos_id, 4, 1, f);
        }
        // tokenizer.ggml.eos_token_id
        else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && vtype == 4) {
            fread(&eos_id, 4, 1, f);
        }
        else if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0 && vtype == 7) {
            uint8_t v = 0;
            fread(&v, 1, 1, f);
            add_bos_ = (v != 0);
        }
        else if (strcmp(key, "tokenizer.ggml.add_eos_token") == 0 && vtype == 7) {
            uint8_t v = 0;
            fread(&v, 1, 1, f);
            add_eos_ = (v != 0);
        }
        else {
            // Skip value based on type
            switch (vtype) {
                case 0: fseek(f, 1, SEEK_CUR); break;   // uint8
                case 1: fseek(f, 1, SEEK_CUR); break;   // int8
                case 2: fseek(f, 2, SEEK_CUR); break;   // uint16
                case 3: fseek(f, 2, SEEK_CUR); break;   // int16
                case 4: fseek(f, 4, SEEK_CUR); break;   // uint32
                case 5: fseek(f, 4, SEEK_CUR); break;   // int32
                case 6: fseek(f, 4, SEEK_CUR); break;   // float32
                case 7: fseek(f, 1, SEEK_CUR); break;   // bool
                case 8: {                                 // string
                    uint64_t slen;
                    fread(&slen, 8, 1, f);
                    fseek(f, slen, SEEK_CUR);
                    break;
                }
                case 9: {                                 // array
                    uint32_t atype;
                    uint64_t alen;
                    fread(&atype, 4, 1, f);
                    fread(&alen, 8, 1, f);
                    if (atype == 8) {  // string array: each is len + bytes
                        for (uint64_t a = 0; a < alen; a++) {
                            uint64_t slen;
                            fread(&slen, 8, 1, f);
                            fseek(f, slen, SEEK_CUR);
                        }
                    } else {
                        // Fixed-width element: size by GGUF type id so bool(7),
                        // int8(0/1), int16(2/3), and 64-bit(10/11/12) arrays are
                        // skipped correctly (a wrong guess here desyncs the whole
                        // metadata stream — e.g. Gemma's bool sliding_window_pattern).
                        int esz = (atype <= 1 || atype == 7) ? 1
                                  : (atype <= 3)            ? 2
                                  : (atype <= 6)            ? 4 : 8;
                        fseek(f, (long)alen * esz, SEEK_CUR);
                    }
                    break;
                }
                case 10: fseek(f, 8, SEEK_CUR); break;  // uint64
                case 11: fseek(f, 8, SEEK_CUR); break;  // int64
                case 12: fseek(f, 8, SEEK_CUR); break;  // float64
                default: fseek(f, 8, SEEK_CUR); break;   // unknown, skip 8
            }
        }
    }

    fclose(f);

    // BUG #8 FIX: build hash map for O(1) token lookup instead of O(V) linear scan
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        token_to_id_[vocab[i]] = i;
    }

    special_tokens_.clear();
    for (int i = 0; i < (int)vocab.size(); i++) {
        const auto& tok = vocab[i];
        if (tok.size() >= 5 &&
            ((tok.rfind("<|", 0) == 0 &&
              tok.compare(tok.size() - 2, 2, "|>") == 0) ||
             tok == "<think>" || tok == "</think>")) {
            special_tokens_.push_back({tok, i});
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    // Build sorted-by-length index for greedy longest-match
    sorted_by_len_.clear();
    sorted_by_len_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        if (!vocab[i].empty())
            sorted_by_len_.push_back(i);
    }
    std::sort(sorted_by_len_.begin(), sorted_by_len_.end(),
              [this](int a, int b) { return vocab[a].size() > vocab[b].size(); });

    // Precompute max token length for early termination
    max_token_len_ = 0;
    for (const auto& t : vocab)
        max_token_len_ = std::max(max_token_len_, (int)t.size());

    fprintf(stderr, "[tokenizer] vocab=%zu, bos=%d(add=%d), eos=%d(add=%d), max_token_len=%d\n",
            vocab.size(), bos_id, add_bos_, eos_id, add_eos_, max_token_len_);
    return !vocab.empty();
}

// SentencePiece/unigram encode: Viterbi over piece scores with byte fallback.
// Matches the GGUF "llama" (SPM) tokenizer used by Gemma. The segment is
// normalized SPM-style (ASCII space -> ▁ U+2581, optional leading ▁) and then
// segmented to maximize the sum of piece scores; an unmatched byte falls back
// to its <0xNN> byte token (kept length-1 so the lattice stays connected).
void Tokenizer::encode_spm(const std::string& text, std::vector<int>& out) const {
    static const std::string kMeta = "\xe2\x96\x81";  // ▁ (U+2581)
    std::string s;
    s.reserve(text.size() + kMeta.size());
    if (add_space_prefix_) s += kMeta;
    for (char c : text) {
        if (c == ' ') s += kMeta;
        else s += c;
    }

    const int n = (int)s.size();
    const float kNeg = -1e30f;
    std::vector<float> best(n + 1, kNeg);
    std::vector<int> back_id(n + 1, -1);
    std::vector<int> back_pos(n + 1, -1);
    best[0] = 0.0f;

    for (int i = 0; i < n; ++i) {
        if (best[i] <= kNeg) continue;
        int max_len = std::min(max_token_len_, n - i);
        for (int len = 1; len <= max_len; ++len) {
            auto it = token_to_id_.find(s.substr(i, len));
            if (it == token_to_id_.end()) continue;
            float sc = best[i] + token_scores_[it->second];
            if (sc > best[i + len]) {
                best[i + len] = sc;
                back_id[i + len] = it->second;
                back_pos[i + len] = i;
            }
        }
        // Byte fallback for the single byte at i.
        char hex[8];
        snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)s[i]);
        auto bf = token_to_id_.find(hex);
        if (bf != token_to_id_.end()) {
            float sc = best[i] + token_scores_[bf->second];
            if (sc > best[i + 1]) {
                best[i + 1] = sc;
                back_id[i + 1] = bf->second;
                back_pos[i + 1] = i;
            }
        }
    }

    if (best[n] <= kNeg) return;  // unreachable (no byte tokens) — skip
    std::vector<int> rev;
    for (int i = n; i > 0; i = back_pos[i]) rev.push_back(back_id[i]);
    out.insert(out.end(), rev.rbegin(), rev.rend());
}

// BUG #8 FIX: O(max_token_len) per position instead of O(V)
// Uses hash map for exact match + early termination by length
std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    if (add_bos_)
        tokens.push_back(bos_id);

    auto encode_plain = [this, &tokens](const std::string& plain) {
        auto pieces = pretokenize_ascii(plain);
        for (const auto& raw_piece : pieces) {
            std::string piece = byte_encode_ ? encode_bytes_gpt2(raw_piece) : raw_piece;

            if (bpe_ranks_.empty()) {
                size_t pos = 0;
                while (pos < piece.size()) {
                    int best_id = -1;
                    size_t best_len = 0;

                    int try_len = std::min(max_token_len_, (int)(piece.size() - pos));
                    for (int len = try_len; len >= 1; len--) {
                        std::string candidate = piece.substr(pos, len);
                        auto it = token_to_id_.find(candidate);
                        if (it != token_to_id_.end()) {
                            best_id = it->second;
                            best_len = len;
                            break;
                        }
                    }

                    if (best_id >= 0) {
                        tokens.push_back(best_id);
                        pos += best_len;
                    } else {
                        int len = utf8_len((unsigned char)piece[pos]);
                        if (pos + len > piece.size()) len = 1;
                        pos += len;
                    }
                }
                continue;
            }

            std::vector<std::string> symbols = split_utf8_symbols(piece);

            while (symbols.size() > 1) {
                int best_rank = INT_MAX;
                size_t best_idx = symbols.size();
                for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                    auto it = bpe_ranks_.find(bpe_key(symbols[i], symbols[i + 1]));
                    if (it != bpe_ranks_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                    }
                }
                if (best_idx == symbols.size()) break;

                symbols[best_idx] += symbols[best_idx + 1];
                symbols.erase(symbols.begin() + best_idx + 1);
            }

            for (const auto& sym : symbols) {
                auto it = token_to_id_.find(sym);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                    continue;
                }

                for (size_t pos = 0; pos < sym.size();) {
                    int len = utf8_len((unsigned char)sym[pos]);
                    if (pos + len > sym.size()) len = 1;
                    std::string fallback = sym.substr(pos, len);
                    auto fb = token_to_id_.find(fallback);
                    if (fb != token_to_id_.end())
                        tokens.push_back(fb->second);
                    pos += len;
                }
            }
        }
    };

    size_t pos = 0;
    while (pos < text.size()) {
        int special_id = -1;
        size_t special_len = 0;
        for (const auto& sp : special_tokens_) {
            if (text.compare(pos, sp.first.size(), sp.first) == 0) {
                special_id = sp.second;
                special_len = sp.first.size();
                break;
            }
        }

        if (special_id >= 0) {
            tokens.push_back(special_id);
            pos += special_len;
            continue;
        }

        size_t next = text.size();
        for (size_t scan = pos + 1; scan < text.size(); ++scan) {
            for (const auto& sp : special_tokens_) {
                if (text.compare(scan, sp.first.size(), sp.first) == 0) {
                    next = scan;
                    goto found_next_special;
                }
            }
        }

found_next_special:
        {
            std::string seg = text.substr(pos, next - pos);
            if (is_spm_ && token_scores_.size() == vocab.size())
                encode_spm(seg, tokens);
            else
                encode_plain(seg);
        }
        pos = next;
    }

    if (add_eos_)
        tokens.push_back(eos_id);

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    if (token_id >= 0 && token_id < (int)vocab.size()) {
        const auto& tok = vocab[token_id];
        // Handle byte tokens like <0x0A> → actual byte
        if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x') {
            unsigned int byte_val;
            if (sscanf(tok.c_str(), "<0x%02X>", &byte_val) == 1)
                return std::string(1, (char)byte_val);
        }
        if (byte_encode_) {
            const auto& dec = byte_decoder();
            std::string out;
            bool decoded_any = false;
            for (size_t pos = 0; pos < tok.size();) {
                int len = utf8_len((unsigned char)tok[pos]);
                if (pos + len > tok.size()) len = 1;
                std::string cp = tok.substr(pos, len);
                auto it = dec.find(cp);
                if (it == dec.end()) {
                    if (!decoded_any) return tok;
                    out += cp;
                } else {
                    out.push_back((char)it->second);
                    decoded_any = true;
                }
                pos += len;
            }
            return out;
        }
        return tok;
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id != bos_id && id != eos_id)
            result += decode(id);
    }
    return result;
}

}  // namespace jllm
