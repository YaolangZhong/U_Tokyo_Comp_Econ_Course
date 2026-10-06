"""Convert a Celsius temperature to Fahrenheit."""
import sys

CELSIUS = 20.0


def to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


if __name__ == "__main__":
    fahrenheit = to_fahrenheit(CELSIUS)
    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    print(f"{CELSIUS:.1f} C = {fahrenheit:.1f} F")
