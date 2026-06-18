// jllm_grammar.h — GBNF grammar-constrained decoding (host-side).
//
// A self-contained port of the llama.cpp GBNF grammar engine: a parser for
// GBNF text and a pushdown-automaton matcher that advances one UTF-8 code
// point at a time. Used to mask sampler logits so the model can only emit
// tokens that continue a valid string under the grammar (issue #86).
//
// Decoupled from the tokenizer on purpose: the matcher operates on a vector
// of per-token decoded byte strings (token id -> literal output bytes), so it
// is trivially unit-testable without a model or GPU.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jllm {

// Grammar element types (mirror llama.cpp's llama_gretype).
enum GreType : uint8_t {
    GRE_END            = 0,  // end of rule definition
    GRE_ALT            = 1,  // start of an alternate definition for a rule
    GRE_RULE_REF       = 2,  // non-terminal: reference to another rule (value = rule id)
    GRE_CHAR           = 3,  // terminal: a single code point (value = code point)
    GRE_CHAR_NOT       = 4,  // inverse char set: [^...]
    GRE_CHAR_RNG_UPPER = 5,  // upper bound of an inclusive range, modifies the preceding CHAR/CHAR_ALT
    GRE_CHAR_ALT       = 6,  // alternate char in a set, modifies the preceding CHAR/RNG_UPPER
    GRE_CHAR_ANY       = 7,  // "." — matches any single code point
};

struct GrammarElement {
    GreType  type;
    uint32_t value;
};

// A parsed grammar: a flat list of rules, each a sequence of elements
// terminated by GRE_END (with GRE_ALT separating alternates).
struct Grammar {
    std::vector<std::vector<GrammarElement>> rules;
    uint32_t start_rule = 0;
    bool     ok         = false;  // false if parsing failed
};

// Parse GBNF `text`, using `root` as the entry rule (default "root").
// Returns a Grammar with ok=false (and logs to stderr) on any error.
Grammar grammar_parse(const std::string& text, const std::string& root = "root");

// Live decoding state: the set of pushdown stacks reachable at the current
// position, plus any partial multi-byte UTF-8 sequence carried across tokens.
struct GrammarState {
    const Grammar* grammar = nullptr;
    std::vector<std::vector<const GrammarElement*>> stacks;
    uint32_t partial_value    = 0;
    int      partial_n_remain = 0;
    bool     done             = false;  // EOS accepted — generation should stop
};

// Initialise `st` from a parsed grammar (computes the initial stacks).
void grammar_state_init(GrammarState& st, const Grammar& g);

// True when the grammar is in an accepting state (some stack is empty),
// i.e. ending the string here is valid and EOS may be emitted.
bool grammar_state_complete(const GrammarState& st);

// Mask `logits` in place: every token whose decoded bytes cannot continue a
// valid string under the current state is set to -INF. `token_bytes[id]` is
// the literal output bytes for token `id`. `eos_id` is kept iff the grammar
// is complete. Returns the number of tokens left un-masked (>=1 if eos kept).
int grammar_apply(GrammarState& st, const std::vector<std::string>& token_bytes,
                  int eos_id, float* logits, int vocab_size);

// Advance the grammar by an accepted token. Returns false if the token does
// not fit the grammar (should not happen after grammar_apply masked it).
// Accepting `eos_id` sets st.done.
bool grammar_accept_token(GrammarState& st, const std::vector<std::string>& token_bytes,
                          int eos_id, int token_id);

}  // namespace jllm
