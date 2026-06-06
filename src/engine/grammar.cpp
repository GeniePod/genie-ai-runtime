// grammar.cpp — GBNF parser + pushdown matcher for constrained decoding.
//
// Port of the llama.cpp GBNF grammar engine (MIT), trimmed to what the
// sampler needs and decoupled from the tokenizer. See jllm_grammar.h and
// issue #86. The matcher advances a set of pushdown stacks one UTF-8 code
// point at a time; a token is allowed iff its decoded bytes can be consumed
// without emptying the stack set.

#include "jllm_grammar.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jllm {

// ── UTF-8 decoding ────────────────────────────────────────────────────────

// Decode a single UTF-8 code point from a grammar-source pointer (used by the
// parser). Returns (code point, pointer past the sequence).
static std::pair<uint32_t, const char*> decode_utf8_char(const char* src) {
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
    uint8_t  first_byte = static_cast<uint8_t>(*src);
    uint8_t  highbits   = first_byte >> 4;
    int      len        = lookup[highbits];
    if (len == 0) len = 1;  // invalid lead byte: consume one byte
    uint8_t  mask  = (1 << (8 - len)) - 1;
    uint32_t value = first_byte & mask;
    const char* end = src + len;  // may overrun on truncated input
    const char* pos = src + 1;
    for (; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

namespace {
struct PartialUtf8 {
    uint32_t value;
    int      n_remain;  // -1 = error, 0 = complete, >0 = bytes still expected
};
}  // namespace

// Decode a byte string (a token's literal bytes) into code points, continuing
// any partial sequence from `start`. The returned vector is 0-terminated.
static std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8_str(
        const std::string& src, PartialUtf8 start) {
    static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
    const char*           pos = src.c_str();
    std::vector<uint32_t> code_points;
    code_points.reserve(src.size() + 1);
    uint32_t value    = start.value;
    int      n_remain = start.n_remain;

    // Finish the carried-over partial sequence, if any.
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), PartialUtf8{0, -1});
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }
    if (start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // Decode the remaining sequences; the last may be incomplete.
    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint8_t highbits   = first_byte >> 4;
        n_remain = lookup[highbits] - 1;
        if (n_remain < 0) {
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), PartialUtf8{0, n_remain});
        }
        uint8_t mask = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;
        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);
    return std::make_pair(std::move(code_points), PartialUtf8{value, n_remain});
}

// ── GBNF parser ───────────────────────────────────────────────────────────

namespace {

struct ParseState {
    std::map<std::string, uint32_t>          symbol_ids;
    std::vector<std::vector<GrammarElement>> rules;
};

bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' ||
           ('0' <= c && c <= '9');
}
bool is_digit_char(char c) { return '0' <= c && c <= '9'; }

uint32_t get_symbol_id(ParseState& state, const char* src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
    auto res = state.symbol_ids.emplace(std::string(src, len), next_id);
    return res.first->second;
}

uint32_t generate_symbol_id(ParseState& state, const std::string& base_name) {
    uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
    state.symbol_ids[base_name + "_" + std::to_string(next_id)] = next_id;
    return next_id;
}

void add_rule(ParseState& state, uint32_t rule_id,
              const std::vector<GrammarElement>& rule) {
    if (state.rules.size() <= rule_id) state.rules.resize(rule_id + 1);
    state.rules[rule_id] = rule;
}

std::pair<uint32_t, const char*> parse_hex(const char* src, int size) {
    const char* pos = src;
    const char* end = src + size;
    uint32_t    value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if ('a' <= c && c <= 'f')      value += c - 'a' + 10;
        else if ('A' <= c && c <= 'F') value += c - 'A' + 10;
        else if ('0' <= c && c <= '9') value += c - '0';
        else break;
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars");
    }
    return std::make_pair(value, pos);
}

const char* parse_space(const char* src, bool newline_ok) {
    const char* pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
           (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') pos++;
        } else {
            pos++;
        }
    }
    return pos;
}

const char* parse_name(const char* src) {
    const char* pos = src;
    while (is_word_char(*pos)) pos++;
    if (pos == src) throw std::runtime_error("expecting name");
    return pos;
}

const char* parse_int(const char* src) {
    const char* pos = src;
    while (is_digit_char(*pos)) pos++;
    if (pos == src) throw std::runtime_error("expecting integer");
    return pos;
}

std::pair<uint32_t, const char*> parse_char(const char* src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair((uint32_t)0x09, src + 2);
            case 'r': return std::make_pair((uint32_t)0x0D, src + 2);
            case 'n': return std::make_pair((uint32_t)0x0A, src + 2);
            case '\\':
            case '"':
            case '[':
            case ']': return std::make_pair((uint32_t)src[1], src + 2);
            default:  throw std::runtime_error("unknown escape");
        }
    } else if (*src) {
        return decode_utf8_char(src);
    }
    throw std::runtime_error("unexpected end of input");
}

const char* parse_alternates(ParseState& state, const char* src,
                             const std::string& rule_name, uint32_t rule_id,
                             bool is_nested);

const char* parse_sequence(ParseState& state, const char* src,
                           const std::string& rule_name,
                           std::vector<GrammarElement>& out, bool is_nested) {
    size_t      last_sym_start = out.size();
    const char* pos = src;

    // Rewrite the last symbol (out[last_sym_start..end]) into a repetition.
    auto handle_repetitions = [&](int min_times, int max_times) {
        if (last_sym_start == out.size()) {
            throw std::runtime_error("expecting preceding item before */+/?/{ ");
        }
        std::vector<GrammarElement> prev(out.begin() + last_sym_start, out.end());
        if (min_times == 0) {
            out.resize(last_sym_start);
        } else {
            for (int i = 1; i < min_times; i++)
                out.insert(out.end(), prev.begin(), prev.end());
        }

        uint32_t last_rec_rule_id = 0;
        int      n_opt = max_times < 0 ? 1 : max_times - min_times;
        std::vector<GrammarElement> rec_rule(prev);
        for (int i = 0; i < n_opt; i++) {
            rec_rule.resize(prev.size());
            uint32_t rec_rule_id = generate_symbol_id(state, rule_name);
            if (i > 0 || max_times < 0) {
                rec_rule.push_back({GRE_RULE_REF,
                                    max_times < 0 ? rec_rule_id : last_rec_rule_id});
            }
            rec_rule.push_back({GRE_ALT, 0});
            rec_rule.push_back({GRE_END, 0});
            add_rule(state, rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) out.push_back({GRE_RULE_REF, last_rec_rule_id});
    };

    while (*pos) {
        if (*pos == '"') {  // literal string
            pos++;
            last_sym_start = out.size();
            while (*pos != '"') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                auto cp = parse_char(pos);
                pos = cp.second;
                out.push_back({GRE_CHAR, cp.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') {  // char class
            pos++;
            GreType start_type = GRE_CHAR;
            if (*pos == '^') { pos++; start_type = GRE_CHAR_NOT; }
            last_sym_start = out.size();
            while (*pos != ']') {
                if (!*pos) throw std::runtime_error("unexpected end of input");
                auto cp = parse_char(pos);
                pos = cp.second;
                GreType type = last_sym_start < out.size() ? GRE_CHAR_ALT : start_type;
                out.push_back({type, cp.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) throw std::runtime_error("unexpected end of input");
                    auto hi = parse_char(pos + 1);
                    pos = hi.second;
                    out.push_back({GRE_CHAR_RNG_UPPER, hi.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) {  // rule reference
            const char* name_end = parse_name(pos);
            uint32_t    ref_id   = get_symbol_id(state, pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = out.size();
            out.push_back({GRE_RULE_REF, ref_id});
        } else if (*pos == '(') {  // grouping
            pos = parse_space(pos + 1, true);
            uint32_t sub_id = generate_symbol_id(state, rule_name);
            pos = parse_alternates(state, pos, rule_name, sub_id, true);
            last_sym_start = out.size();
            out.push_back({GRE_RULE_REF, sub_id});
            if (*pos != ')') throw std::runtime_error("expecting ')'");
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') {  // any char
            last_sym_start = out.size();
            out.push_back({GRE_CHAR_ANY, 0});
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, -1);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, -1);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);
            if (!is_digit_char(*pos)) throw std::runtime_error("expecting int in {}");
            const char* int_end = parse_int(pos);
            int min_times = std::stoi(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);
            int max_times = -1;
            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);
                if (is_digit_char(*pos)) {
                    const char* e = parse_int(pos);
                    max_times = std::stoi(std::string(pos, e - pos));
                    pos = parse_space(e, is_nested);
                }
                if (*pos != '}') throw std::runtime_error("expecting '}'");
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error("expecting ',' or '}'");
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char* parse_alternates(ParseState& state, const char* src,
                             const std::string& rule_name, uint32_t rule_id,
                             bool is_nested) {
    std::vector<GrammarElement> rule;
    const char* pos = parse_sequence(state, src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({GRE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(state, pos, rule_name, rule, is_nested);
    }
    rule.push_back({GRE_END, 0});
    add_rule(state, rule_id, rule);
    return pos;
}

const char* parse_rule(ParseState& state, const char* src) {
    const char* name_end = parse_name(src);
    const char* pos      = parse_space(name_end, false);
    size_t      name_len = name_end - src;
    uint32_t    rule_id  = get_symbol_id(state, src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error("expecting ::=");
    }
    pos = parse_space(pos + 3, true);
    pos = parse_alternates(state, pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error("expecting newline or end");
    }
    return parse_space(pos, true);
}

}  // namespace

Grammar grammar_parse(const std::string& text, const std::string& root) {
    Grammar g;
    ParseState state;
    try {
        const char* pos = parse_space(text.c_str(), true);
        while (*pos) pos = parse_rule(state, pos);

        // Every rule reference must resolve to a defined rule.
        for (const auto& rule : state.rules) {
            for (const auto& el : rule) {
                if (el.type == GRE_RULE_REF &&
                    (el.value >= state.rules.size() || state.rules[el.value].empty())) {
                    throw std::runtime_error("undefined rule reference");
                }
            }
        }
        auto it = state.symbol_ids.find(root);
        if (it == state.symbol_ids.end()) {
            throw std::runtime_error("root rule '" + root + "' not found");
        }
        g.rules      = std::move(state.rules);
        g.start_rule = it->second;
        g.ok         = true;
    } catch (const std::exception& err) {
        fprintf(stderr, "[grammar] parse error: %s\n", err.what());
        g.ok = false;
    }
    return g;
}

// ── Pushdown matcher ──────────────────────────────────────────────────────

namespace {

using Stack  = std::vector<const GrammarElement*>;
using Stacks = std::vector<Stack>;

bool is_end_of_seq(const GrammarElement* pos) {
    return pos->type == GRE_END || pos->type == GRE_ALT;
}

// Does `chr` satisfy the char-set element(s) starting at `pos`? Returns the
// match result and the pointer just past the char-set.
std::pair<bool, const GrammarElement*> match_char(const GrammarElement* pos,
                                                  uint32_t chr) {
    bool found = false;
    bool is_positive = pos->type == GRE_CHAR || pos->type == GRE_CHAR_ANY;
    do {
        if (pos[1].type == GRE_CHAR_RNG_UPPER) {
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == GRE_CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == GRE_CHAR_ALT);
    return std::make_pair(found == is_positive, pos);
}

// Expand a stack so its top is a terminal (char element) or empty, pushing all
// reachable stacks into `out`.
void advance_stack(const std::vector<std::vector<GrammarElement>>& rules,
                   const Stack& stack, Stacks& out) {
    if (stack.empty()) {
        if (std::find(out.begin(), out.end(), stack) == out.end()) out.push_back(stack);
        return;
    }
    const GrammarElement* pos = stack.back();
    switch (pos->type) {
        case GRE_RULE_REF: {
            const size_t          rule_id = pos->value;
            const GrammarElement* subpos  = rules[rule_id].data();
            do {
                Stack new_stack(stack.begin(), stack.end() - 1);
                if (!is_end_of_seq(pos + 1)) new_stack.push_back(pos + 1);
                if (!is_end_of_seq(subpos))  new_stack.push_back(subpos);
                advance_stack(rules, new_stack, out);
                while (!is_end_of_seq(subpos)) subpos++;
                if (subpos->type == GRE_ALT) subpos++;
                else break;
            } while (true);
            break;
        }
        case GRE_CHAR:
        case GRE_CHAR_NOT:
        case GRE_CHAR_ANY:
            if (std::find(out.begin(), out.end(), stack) == out.end()) out.push_back(stack);
            break;
        default:
            break;  // unreachable for well-formed stacks
    }
}

// Advance every stack by one code point.
Stacks accept_char(const std::vector<std::vector<GrammarElement>>& rules,
                   const Stacks& stacks, uint32_t chr) {
    Stacks out;
    for (const auto& stack : stacks) {
        if (stack.empty()) continue;
        auto m = match_char(stack.back(), chr);
        if (!m.first) continue;
        const GrammarElement* pos = m.second;
        Stack new_stack(stack.begin(), stack.end() - 1);
        if (!is_end_of_seq(pos)) new_stack.push_back(pos);
        advance_stack(rules, new_stack, out);
    }
    return out;
}

// Can `bytes` be consumed from `stacks` without emptying the set? `partial` is
// updated to reflect any trailing incomplete UTF-8 sequence. A trailing
// partial sequence is treated as acceptable (validated when completed).
bool stacks_accept_bytes(const Grammar& g, Stacks stacks, const std::string& bytes,
                         PartialUtf8 partial, Stacks* result, PartialUtf8* out_partial) {
    auto decoded = decode_utf8_str(bytes, partial);
    const std::vector<uint32_t>& cps = decoded.first;
    if (decoded.second.n_remain < 0) return false;  // invalid UTF-8

    for (size_t i = 0; i + 1 < cps.size(); i++) {  // last entry is the 0 sentinel
        stacks = accept_char(g.rules, stacks, cps[i]);
        if (stacks.empty()) return false;
    }
    if (result)      *result = std::move(stacks);
    if (out_partial) *out_partial = decoded.second;
    return true;
}

}  // namespace

void grammar_state_init(GrammarState& st, const Grammar& g) {
    st.grammar          = &g;
    st.partial_value    = 0;
    st.partial_n_remain = 0;
    st.done             = false;
    st.stacks.clear();
    if (!g.ok || g.rules.empty()) return;

    const GrammarElement* pos = g.rules[g.start_rule].data();
    do {
        Stack stack;
        if (!is_end_of_seq(pos)) stack.push_back(pos);
        advance_stack(g.rules, stack, st.stacks);
        while (!is_end_of_seq(pos)) pos++;
        if (pos->type == GRE_ALT) pos++;
        else break;
    } while (true);
}

bool grammar_state_complete(const GrammarState& st) {
    for (const auto& s : st.stacks)
        if (s.empty()) return true;
    return false;
}

int grammar_apply(GrammarState& st, const std::vector<std::string>& token_bytes,
                  int eos_id, float* logits, int vocab_size) {
    if (!st.grammar || !st.grammar->ok) return vocab_size;

    const PartialUtf8 partial{st.partial_value, st.partial_n_remain};
    const bool complete = grammar_state_complete(st);
    int allowed = 0;

    for (int id = 0; id < vocab_size; id++) {
        if (!std::isfinite(logits[id])) continue;  // already masked / non-finite
        if (id == eos_id) {
            if (complete) allowed++;
            else          logits[id] = -INFINITY;
            continue;
        }
        if (id >= 0 && id < (int)token_bytes.size() &&
            stacks_accept_bytes(*st.grammar, st.stacks, token_bytes[id], partial,
                                nullptr, nullptr)) {
            allowed++;
        } else {
            logits[id] = -INFINITY;
        }
    }

    // Safety: never mask the whole vocab. If nothing survives, free EOS so the
    // sampler can terminate instead of falling back to token 0.
    if (allowed == 0 && eos_id >= 0 && eos_id < vocab_size) {
        logits[eos_id] = 0.0f;
        allowed = 1;
    }
    return allowed;
}

bool grammar_accept_token(GrammarState& st, const std::vector<std::string>& token_bytes,
                          int eos_id, int token_id) {
    if (!st.grammar || !st.grammar->ok) return true;
    if (token_id == eos_id) {
        st.done = true;
        return true;
    }
    if (token_id < 0 || token_id >= (int)token_bytes.size()) return false;

    Stacks      next;
    PartialUtf8 out_partial{0, 0};
    const PartialUtf8 partial{st.partial_value, st.partial_n_remain};
    if (!stacks_accept_bytes(*st.grammar, st.stacks, token_bytes[token_id], partial,
                             &next, &out_partial)) {
        return false;
    }
    st.stacks           = std::move(next);
    st.partial_value    = out_partial.value;
    st.partial_n_remain = out_partial.n_remain;
    return true;
}

}  // namespace jllm
