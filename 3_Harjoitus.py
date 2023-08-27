def bin_to_num():
    binary = input("give 4 bytes binary number: ")

    if len(binary) == 4:
        try:
            print(int(binary,2))
            bin_to_num()
        except:
            print("i said 4 letter binary number")
            bin_to_num()

    else:
        print("i said 4 bytes")
        bin_to_num()
bin_to_num()
