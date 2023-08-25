binary = input("give 4 bytes binary number: ")

if len(binary) == 4:
    try:
        print(int(binary,2))
    except:
        print("i said 4 letter binary number")

else:
    print("i said 4 bytes")
