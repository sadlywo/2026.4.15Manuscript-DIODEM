from pathlib import Path

Import("env")

xcubeai_root = Path(env.GetProjectOption("custom_xcubeai_root"))
runtime_lib = (
    xcubeai_root
    / "Middlewares"
    / "ST"
    / "AI"
    / "Lib"
    / "GCC"
    / "STM32H7"
    / "NetworkRuntime1010_CM7_GCC.a"
)

if not runtime_lib.exists():
    raise FileNotFoundError(f"X-CUBE-AI runtime library not found: {runtime_lib}")

env.Append(
    LINKFLAGS=[
        "-mfpu=fpv5-d16",
        "-mfloat-abi=hard",
        "-Wl,--whole-archive",
        str(runtime_lib),
        "-Wl,--no-whole-archive",
    ]
)
