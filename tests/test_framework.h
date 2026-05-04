// Tiny test framework — header-only, no deps. TEST(name) registers a test;
// EXPECT_NEAR / EXPECT_TRUE throw on failure; main() calls test::run_all().

#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

struct Test { std::string name; std::function<void()> fn; };
inline std::vector<Test>& registry() { static std::vector<Test> r; return r; }

struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, f}); } };

#define TEST(name)                                                          \
    static void test_##name();                                              \
    static test::Reg reg_##name(#name, test_##name);                        \
    static void test_##name()

class Failure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#define EXPECT_NEAR(a, b, tol) do {                                         \
    double _a = (double)(a), _b = (double)(b), _t = (double)(tol);          \
    if (std::isnan(_a) || std::isnan(_b) || std::abs(_a - _b) > _t)         \
        throw test::Failure(std::string(__FILE__ ":") + std::to_string(__LINE__) \
            + " EXPECT_NEAR(" #a ", " #b ", " #tol ")  got " + std::to_string(_a) \
            + ", expected " + std::to_string(_b));                          \
} while (0)

#define EXPECT_TRUE(cond) do {                                              \
    if (!(cond))                                                            \
        throw test::Failure(std::string(__FILE__ ":") + std::to_string(__LINE__) \
            + " EXPECT_TRUE(" #cond ")");                                   \
} while (0)

inline int run_all() {
    int pass = 0, fail = 0;
    for (auto& t : registry()) {
        try {
            t.fn();
            std::cout << "  \033[32mPASS\033[0m " << t.name << "\n";
            ++pass;
        } catch (const Failure& e) {
            std::cout << "  \033[31mFAIL\033[0m " << t.name << "\n         " << e.what() << "\n";
            ++fail;
        } catch (const std::exception& e) {
            std::cout << "  \033[31mFAIL\033[0m " << t.name << " (uncaught)\n         " << e.what() << "\n";
            ++fail;
        }
    }
    std::cout << "\n" << pass << "/" << (pass + fail) << " passed\n";
    return fail == 0 ? 0 : 1;
}

} // namespace test
