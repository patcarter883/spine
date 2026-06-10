#include <string>

namespace geo {

enum class Color { Red, Green, Blue };

struct Point {
    int x;
    int y;
};

class Greeter {
public:
    explicit Greeter(std::string name) : name_(std::move(name)) {}
    ~Greeter() {}

    std::string greet() const {
        return "hello " + name_;
    }

    std::string shout() const;

private:
    std::string name_;
};

std::string Greeter::shout() const {
    return "HELLO " + name_;
}

double dot(const Point& a, const Point& b) {
    return a.x * b.x + a.y * b.y;
}

const std::string& pick(const std::string& a) {
    return a;
}

int *make_counter() {
    return new int(0);
}

}  // namespace geo
