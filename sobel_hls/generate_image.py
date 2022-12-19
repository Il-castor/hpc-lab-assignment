import random
l = "uint8_t input[512*512] = {\n"
for i in range((512*512)-1):
    l += str(random.randint(0,255))
    l += ",\n"
l += str(random.randint(0,255))
l += "};\n"

print(l)