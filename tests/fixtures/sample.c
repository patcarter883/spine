#include <stdio.h>

struct Point {
    int x;
    int y;
};

union Value {
    int i;
    float f;
};

enum Color {
    RED,
    GREEN,
    BLUE,
};

int add(int a, int b) {
    return a + b;
}

static char *format_point(struct Point p) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "(%d, %d)", p.x, p.y);
    return buf;
}
