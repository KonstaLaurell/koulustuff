import random
lotto_rivi = input("anna 7 numeroa 1-40 : ")
lotto_rivi.split()
print(lotto_rivi)
if isinstance(lotto_rivi, list):
    print("on lista")
if len(lotto_rivi) == 7:
    for i in lotto_rivi:
        try: 
            i = int(i)
        except: 
            print("error ei toimi.exe")
            print(ValueError)
        
else:
    print("not 7 lenght")