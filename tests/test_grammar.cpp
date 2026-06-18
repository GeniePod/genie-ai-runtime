// test_grammar.cpp — host unit test for the GBNF engine (issue #86).
//
// Pure CPU; no model or GPU. Builds a synthetic token table and drives the
// matcher to verify masking (which tokens are allowed) and acceptance
// (state advances correctly) across enums, char classes, repetition, and a
// JSON-object grammar with rule references.

#include "jllm_grammar.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace jllm;

static int g_failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                               \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            g_failures++;                                            \
        }                                                            \
    } while (0)

// Synthetic vocab: ids 0..127 are single ASCII bytes; a few multi-char tokens
// above that; eos at 200.
static const int kEos = 200;
static std::vector<std::string> make_token_table() {
    std::vector<std::string> t(256);
    for (int i = 0; i < 128; i++) t[i] = std::string(1, (char)i);
    t[130] = "turn_on";
    t[131] = "turn_off";
    t[132] = "turn";
    t[133] = "_on";
    t[134] = "xyz";
    t[kEos] = "<eos>";
    return t;
}

// Is token `id` allowed in the current state? (does grammar_apply keep it finite)
static bool allowed(GrammarState& st, const std::vector<std::string>& toks, int id) {
    std::vector<float> logits(256, 0.0f);
    grammar_apply(st, toks, kEos, logits.data(), (int)logits.size());
    return std::isfinite(logits[id]);
}

// Feed a UTF-8 string one single-char token at a time, asserting each char is
// allowed and advances the state.
static void feed_str(GrammarState& st, const std::vector<std::string>& toks,
                     const std::string& s) {
    for (unsigned char c : s) {
        if (!allowed(st, toks, (int)c)) {
            fprintf(stderr, "FAIL: char '%c' (0x%02x) masked but expected allowed\n", c, c);
            g_failures++;
            return;
        }
        bool ok = grammar_accept_token(st, toks, kEos, (int)c);
        if (!ok) {
            fprintf(stderr, "FAIL: accept of '%c' rejected\n", c);
            g_failures++;
            return;
        }
    }
}

static void test_enum() {
    auto toks = make_token_table();
    Grammar g = grammar_parse("root ::= \"turn_on\" | \"turn_off\"\n");
    CHECK(g.ok, "enum grammar parses");

    GrammarState st;
    grammar_state_init(st, g);
    CHECK(!grammar_state_complete(st), "enum not complete at start");

    // Single-char starts: 't' allowed; '_' and 'x' not.
    CHECK(allowed(st, toks, 't'), "enum: 't' allowed at start");
    CHECK(!allowed(st, toks, '_'), "enum: '_' masked at start");
    CHECK(!allowed(st, toks, 'x'), "enum: 'x' masked at start");
    // Multi-char tokens: turn_on / turn_off / turn allowed; xyz not.
    CHECK(allowed(st, toks, 130), "enum: 'turn_on' token allowed");
    CHECK(allowed(st, toks, 131), "enum: 'turn_off' token allowed");
    CHECK(allowed(st, toks, 132), "enum: 'turn' prefix token allowed");
    CHECK(!allowed(st, toks, 134), "enum: 'xyz' token masked");
    CHECK(!allowed(st, toks, kEos), "enum: EOS masked before complete");

    // Drive "turn" then "_off" via multi-char tokens.
    CHECK(grammar_accept_token(st, toks, kEos, 132), "accept 'turn'");
    CHECK(allowed(st, toks, 133), "enum: '_on' allowed after 'turn'");
    CHECK(!allowed(st, toks, 't'), "enum: 't' masked after 'turn'");
    CHECK(grammar_accept_token(st, toks, kEos, 133), "accept '_on'");
    CHECK(grammar_state_complete(st), "enum complete after turn_on");
    CHECK(allowed(st, toks, kEos), "enum: EOS allowed when complete");
}

static void test_charclass_repetition() {
    auto toks = make_token_table();
    Grammar g = grammar_parse("root ::= [a-z]+\n");
    CHECK(g.ok, "charclass grammar parses");

    GrammarState st;
    grammar_state_init(st, g);
    CHECK(!grammar_state_complete(st), "[a-z]+ not complete at start (needs >=1)");
    CHECK(allowed(st, toks, 'a'), "[a-z]+: 'a' allowed");
    CHECK(allowed(st, toks, 'z'), "[a-z]+: 'z' allowed");
    CHECK(!allowed(st, toks, '1'), "[a-z]+: '1' masked");
    CHECK(!allowed(st, toks, 'Z'), "[a-z]+: uppercase masked");

    feed_str(st, toks, "abc");
    CHECK(grammar_state_complete(st), "[a-z]+ complete after 'abc'");
    CHECK(allowed(st, toks, 'a'), "[a-z]+: still allows more after 'abc'");
    CHECK(allowed(st, toks, kEos), "[a-z]+: EOS allowed after 'abc'");
}

static void test_json_object() {
    auto toks = make_token_table();
    // A flat tool-call JSON: {"action":"<enum>"} with optional spaces.
    const char* gbnf =
        "root   ::= \"{\" ws \"\\\"action\\\"\" ws \":\" ws \"\\\"\" action \"\\\"\" ws \"}\"\n"
        "action ::= \"turn_on\" | \"turn_off\"\n"
        "ws     ::= [ ]*\n";
    Grammar g = grammar_parse(gbnf);
    CHECK(g.ok, "json grammar parses");

    GrammarState st;
    grammar_state_init(st, g);
    CHECK(allowed(st, toks, '{'), "json: '{' allowed at start");
    CHECK(!allowed(st, toks, '}'), "json: '}' masked at start");

    // Walk a valid document; every required char must be allowed.
    feed_str(st, toks, "{\"action\":\"");
    // At the value position, the enum applies: 'turn_on' token allowed, 'xyz' not.
    CHECK(allowed(st, toks, 130), "json: 'turn_on' allowed at value");
    CHECK(!allowed(st, toks, 134), "json: 'xyz' masked at value");
    CHECK(!allowed(st, toks, kEos), "json: EOS masked mid-document");

    feed_str(st, toks, "turn_on\"}");
    CHECK(grammar_state_complete(st), "json: complete after full document");
    CHECK(allowed(st, toks, kEos), "json: EOS allowed when document complete");

    // A second, space-tolerant document parses too.
    GrammarState st2;
    grammar_state_init(st2, g);
    feed_str(st2, toks, "{ \"action\" : \"turn_off\" }");
    CHECK(grammar_state_complete(st2), "json: space-tolerant document complete");
}

static void test_bad_grammar() {
    Grammar g = grammar_parse("root ::= undefined_rule\n");
    CHECK(!g.ok, "undefined rule reference fails to parse");
    Grammar g2 = grammar_parse("notroot ::= \"a\"\n");
    CHECK(!g2.ok, "missing root rule fails to parse");
}

int main() {
    test_enum();
    test_charclass_repetition();
    test_json_object();
    test_bad_grammar();
    if (g_failures == 0) {
        printf("test_grammar: ALL PASS\n");
        return 0;
    }
    printf("test_grammar: %d FAILURE(S)\n", g_failures);
    return 1;
}
