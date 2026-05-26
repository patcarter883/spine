export function greet(name: string): string {
  return `Hello, ${name}`;
}

export const shout = (name: string): string => `HELLO ${name.toUpperCase()}`;

export class Greeter {
  constructor(private prefix: string) {}

  say(name: string): string {
    return `${this.prefix} ${name}`;
  }
}

export interface Speaker {
  speak(): string;
}
