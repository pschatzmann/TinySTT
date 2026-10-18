#pragma once
// Minimal test harness (no external framework, like TinyTTS): TEST() registers
// a function, CHECK*() record failures, runAllTests() returns the exit code.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace testutil {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}
inline int& failures() {
  static int f = 0;
  return f;
}
inline int& checks() {
  static int c = 0;
  return c;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, fn}); }
};

inline void fail(const char* file, int line, const std::string& msg) {
  failures()++;
  std::printf("  FAIL %s:%d: %s\n", file, line, msg.c_str());
}

/// Runs all tests, or only those whose name contains $TINYSTT_TEST_FILTER
inline int runAllTests() {
  const char* filter = std::getenv("TINYSTT_TEST_FILTER");
  int run = 0, failed_tests = 0;
  for (auto& t : registry()) {
    if (filter && std::string(t.name).find(filter) == std::string::npos) continue;
    int before = failures();
    std::printf("[ RUN  ] %s\n", t.name);
    std::fflush(stdout);
    t.fn();
    bool ok = failures() == before;
    std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
    run++;
    if (!ok) failed_tests++;
  }
  std::printf("%d tests, %d checks, %d failed tests\n", run, checks(), failed_tests);
  return failed_tests == 0 && run > 0 ? 0 : 1;
}

}  // namespace testutil

#define TU_CAT2(a, b) a##b
#define TU_CAT(a, b) TU_CAT2(a, b)
#define TEST(name)                                                    \
  static void name();                                                 \
  static testutil::Registrar TU_CAT(reg_, name)(#name, name);         \
  static void name()

#define CHECK(cond)                                                   \
  do {                                                                \
    testutil::checks()++;                                             \
    if (!(cond)) testutil::fail(__FILE__, __LINE__, #cond);           \
  } while (0)

#define CHECK_EQ(a, b)                                                \
  do {                                                                \
    testutil::checks()++;                                             \
    auto _va = (a);                                                   \
    auto _vb = (b);                                                   \
    if (!(_va == _vb))                                                \
      testutil::fail(__FILE__, __LINE__,                              \
                     std::string(#a " == " #b " (") +                 \
                         std::to_string((long long)_va) + " vs " +   \
                         std::to_string((long long)_vb) + ")");       \
  } while (0)

#define CHECK_STR(a, b)                                               \
  do {                                                                \
    testutil::checks()++;                                             \
    std::string _sa = (a), _sb = (b);                                 \
    if (_sa != _sb)                                                   \
      testutil::fail(__FILE__, __LINE__,                              \
                     "\"" + _sa + "\" != \"" + _sb + "\"");           \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                         \
  do {                                                                \
    testutil::checks()++;                                             \
    double _va = (a), _vb = (b);                                      \
    if (!(std::fabs(_va - _vb) <= (tol)))                             \
      testutil::fail(__FILE__, __LINE__,                              \
                     std::string(#a " ~= " #b " (") +                 \
                         std::to_string(_va) + " vs " +               \
                         std::to_string(_vb) + ")");                  \
  } while (0)

// No argc/argv: the Arduino API's Common.h declares `int main()` (weak).
#define TEST_MAIN() \
  int main() { return testutil::runAllTests(); }
