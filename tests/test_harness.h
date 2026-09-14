#ifndef RT_TEST_HARNESS_H
#define RT_TEST_HARNESS_H

// A zero-dependency test harness.
//
// Deliberately has no CMake FetchContent, no submodule, and no network access:
// it must compile with nvcc (for the CUDA branch) and with a plain host C++17
// compiler (for the serial branch) without any configuration difference.
//
// Checks are non-fatal by default -- a failing check records the failure and
// lets the rest of the test body run, so one broken invariant does not hide the
// five behind it. Use RT_REQUIRE when continuing would be meaningless (e.g. a
// null pointer that the next line dereferences).
//
//   TEST(sphere, hit_from_outside) {
//       RT_CHECK_NEAR(rec.t, 4.0f, 1e-5f);
//       RT_CHECK_VEC (rec.normal, 0.0f, 0.0f, -1.0f, 1e-5f);
//   }
//
//   int main(int argc, char** argv) { return rt_test::run_all(argc, argv); }

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rt_test {

struct Case {
    const char* suite;
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

// Per-test mutable state. Globals rather than a context object so the check
// macros stay usable inside plain helper functions called from a test body.
inline int& failures_in_case() { static int n = 0; return n; }
inline int& checks_in_case()   { static int n = 0; return n; }
inline bool& aborted_case()    { static bool b = false; return b; }

struct Registrar {
    Registrar(const char* suite, const char* name, void (*fn)()) {
        registry().push_back(Case{suite, name, fn});
    }
};

// True while the "  case_name ..." status line is still open and waiting for
// its result, so the first failure knows to break the line before printing.
inline bool& line_open() { static bool b = false; return b; }

inline void record_pass() { ++checks_in_case(); }

inline void record_failure(const char* file, int line, const std::string& what) {
    ++checks_in_case();
    ++failures_in_case();
    if (line_open()) { std::printf("\n"); line_open() = false; }
    std::printf("      x %s\n        at %s:%d\n", what.c_str(), file, line);
}

// ---------------------------------------------------------------- formatting

inline std::string fmt(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

inline std::string fmt(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    return buf;
}

// ---------------------------------------------------------------- predicates

// NaN-safe: a NaN operand always fails rather than silently passing through
// the `<=` comparison, which is exactly the bug this is most likely to catch.
inline bool near(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::fabs(a - b) <= tol;
}

// Relative tolerance, for quantities whose magnitude varies a lot across
// scenes (scene-space t values in the Cornell box run to ~1000, in the
// sphere scenes to ~5).
inline bool near_rel(double a, double b, double rel_tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= rel_tol * scale;
}

// -------------------------------------------------------------------- macros

#define RT_CONCAT_(a, b) a##b
#define RT_CONCAT(a, b) RT_CONCAT_(a, b)

#define TEST(suite_, name_)                                                    \
    static void RT_CONCAT(rt_case_, __LINE__)();                               \
    static ::rt_test::Registrar RT_CONCAT(rt_reg_, __LINE__)(                  \
        #suite_, #name_, &RT_CONCAT(rt_case_, __LINE__));                      \
    static void RT_CONCAT(rt_case_, __LINE__)()

#define RT_CHECK(expr)                                                         \
    do {                                                                       \
        if (expr) { ::rt_test::record_pass(); }                                \
        else { ::rt_test::record_failure(__FILE__, __LINE__,                   \
                   std::string("expected true: ") + #expr); }                  \
    } while (0)

#define RT_CHECK_FALSE(expr)                                                   \
    do {                                                                       \
        if (!(expr)) { ::rt_test::record_pass(); }                             \
        else { ::rt_test::record_failure(__FILE__, __LINE__,                   \
                   std::string("expected false: ") + #expr); }                 \
    } while (0)

#define RT_CHECK_EQ(a, b)                                                      \
    do {                                                                       \
        const long long rt_a_ = (long long)(a);                                \
        const long long rt_b_ = (long long)(b);                                \
        if (rt_a_ == rt_b_) { ::rt_test::record_pass(); }                      \
        else { ::rt_test::record_failure(__FILE__, __LINE__,                   \
                   std::string(#a " == " #b "  (got ") + ::rt_test::fmt(rt_a_) \
                   + ", want " + ::rt_test::fmt(rt_b_) + ")"); }               \
    } while (0)

#define RT_CHECK_NEAR(a, b, tol)                                               \
    do {                                                                       \
        const double rt_a_ = (double)(a);                                      \
        const double rt_b_ = (double)(b);                                      \
        if (::rt_test::near(rt_a_, rt_b_, (double)(tol))) {                    \
            ::rt_test::record_pass();                                          \
        } else {                                                               \
            ::rt_test::record_failure(__FILE__, __LINE__,                      \
                std::string(#a "  (got ") + ::rt_test::fmt(rt_a_)              \
                + ", want " + ::rt_test::fmt(rt_b_)                            \
                + " +/- " + ::rt_test::fmt((double)(tol)) + ")");              \
        }                                                                      \
    } while (0)

#define RT_CHECK_NEAR_REL(a, b, rel)                                           \
    do {                                                                       \
        const double rt_a_ = (double)(a);                                      \
        const double rt_b_ = (double)(b);                                      \
        if (::rt_test::near_rel(rt_a_, rt_b_, (double)(rel))) {                \
            ::rt_test::record_pass();                                          \
        } else {                                                               \
            ::rt_test::record_failure(__FILE__, __LINE__,                      \
                std::string(#a "  (got ") + ::rt_test::fmt(rt_a_)              \
                + ", want " + ::rt_test::fmt(rt_b_)                            \
                + " rel " + ::rt_test::fmt((double)(rel)) + ")");              \
        }                                                                      \
    } while (0)

// Works with any type exposing x()/y()/z() -- the CUDA vec3 and the serial
// branch's vec3 both qualify, so vector assertions port between branches.
#define RT_CHECK_VEC(v, ex, ey, ez, tol)                                       \
    do {                                                                       \
        const auto& rt_v_ = (v);                                               \
        const bool rt_ok_ = ::rt_test::near((double)rt_v_.x(), (double)(ex), (double)(tol)) \
                         && ::rt_test::near((double)rt_v_.y(), (double)(ey), (double)(tol)) \
                         && ::rt_test::near((double)rt_v_.z(), (double)(ez), (double)(tol)); \
        if (rt_ok_) { ::rt_test::record_pass(); }                              \
        else {                                                                 \
            ::rt_test::record_failure(__FILE__, __LINE__,                      \
                std::string(#v "  (got ")                                      \
                + ::rt_test::fmt((double)rt_v_.x()) + ", "                     \
                + ::rt_test::fmt((double)rt_v_.y()) + ", "                     \
                + ::rt_test::fmt((double)rt_v_.z()) + "; want "                \
                + ::rt_test::fmt((double)(ex)) + ", "                          \
                + ::rt_test::fmt((double)(ey)) + ", "                          \
                + ::rt_test::fmt((double)(ez))                                 \
                + " +/- " + ::rt_test::fmt((double)(tol)) + ")");              \
        }                                                                      \
    } while (0)

// Fatal: records the failure and abandons the rest of this test case.
#define RT_REQUIRE(expr)                                                       \
    do {                                                                       \
        if (expr) { ::rt_test::record_pass(); }                                \
        else {                                                                 \
            ::rt_test::record_failure(__FILE__, __LINE__,                      \
                std::string("REQUIRED: ") + #expr);                            \
            ::rt_test::aborted_case() = true;                                  \
            return;                                                            \
        }                                                                      \
    } while (0)

#define RT_FAIL(msg)                                                           \
    ::rt_test::record_failure(__FILE__, __LINE__, std::string(msg))

// ---------------------------------------------------------------- the runner

// argv[1..] are case-insensitive substring filters matched against
// "suite.name"; with no filters, everything runs.
inline int run_all(int argc, char** argv) {
    std::vector<std::string> filters;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '-') filters.push_back(argv[i]);
    }

    auto lower = [](std::string s) {
        for (char& c : s) c = (char)std::tolower((unsigned char)c);
        return s;
    };

    int ran = 0, failed_cases = 0, total_checks = 0, total_failures = 0;
    std::string current_suite;

    for (const Case& c : registry()) {
        const std::string full = std::string(c.suite) + "." + c.name;
        if (!filters.empty()) {
            bool match = false;
            for (const std::string& f : filters) {
                if (lower(full).find(lower(f)) != std::string::npos) { match = true; break; }
            }
            if (!match) continue;
        }

        if (current_suite != c.suite) {
            current_suite = c.suite;
            std::printf("\n[%s]\n", c.suite);
        }

        failures_in_case() = 0;
        checks_in_case()   = 0;
        aborted_case()     = false;

        // The name is flushed *before* the body runs so that a device-side trap
        // or a hard crash still tells you which case died.
        std::printf("  %-48s", c.name);
        line_open() = true;
        std::fflush(stdout);
        c.fn();

        ++ran;
        total_checks   += checks_in_case();
        total_failures += failures_in_case();

        if (failures_in_case() == 0) {
            std::printf("ok  (%d checks)\n", checks_in_case());
        } else {
            ++failed_cases;
            if (line_open()) std::printf("\n");
            std::printf("  %-48sFAILED (%d/%d checks)%s\n", c.name,
                        failures_in_case(), checks_in_case(),
                        aborted_case() ? " [aborted early]" : "");
        }
        line_open() = false;
    }

    std::printf("\n----------------------------------------------------------\n");
    if (total_failures == 0) {
        std::printf("PASS  %d cases, %d checks\n", ran, total_checks);
        return 0;
    }
    std::printf("FAIL  %d/%d cases, %d/%d checks failed\n",
                failed_cases, ran, total_failures, total_checks);
    return 1;
}

} // namespace rt_test

#endif // RT_TEST_HARNESS_H
