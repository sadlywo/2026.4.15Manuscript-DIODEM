from SCons.Script import Import


Import("env")

env.Append(LINKFLAGS=["-mfpu=fpv5-d16", "-mfloat-abi=hard"])
