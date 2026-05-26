<?php

function greet(string $name): string {
    return "Hello, " . $name;
}

class Greeter {
    private string $prefix;

    public function __construct(string $prefix) {
        $this->prefix = $prefix;
    }

    public function say(string $name): string {
        return $this->prefix . " " . $name;
    }
}

interface Speaker {
    public function speak(): string;
}
