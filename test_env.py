import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test if key packages are available
try:
    import pandas

    print("✓ pandas is available")
except ImportError:
    print("✗ pandas not available")
