import random
def input_lotto_numbers():
    while True:
        lotto_rivi = input("Anna 7 numeroa 1-40 erotettuna välilyönnillä: ").split()
        if len(lotto_rivi) != 7:
            print("Syötä tarkalleen 7 numeroa.")
            continue
        try:
            lotto_rivi = [int(num) for num in lotto_rivi]
            if all(1 <= num <= 40 for num in lotto_rivi):
                return lotto_rivi
            else:
                print("Kaikkien numeroiden tulee olla välillä 1-40.")
        except ValueError:
            print("Virheellinen syöte. Kaikkien numeroiden tulee olla kokonaislukuja.")
lotto_rivi = input_lotto_numbers()
montako = input("monta kertaa numerot arvotaan: ")
print(lotto_rivi)
montako = int(montako)
nelja_oikein = 0
viisi_oikein = 0
kuusi_oikein = 0
seitseman_oikein = 0
monta_kertaa = 0


if len(lotto_rivi) == 7:
    for i in range(montako):
        voittavatnumerot = []

        while len(voittavatnumerot) < 7: 

            uusinumero = random.randint(1,40)
            if uusinumero not in voittavatnumerot:
                voittavatnumerot.append(uusinumero)
        oikein_menneet = set(voittavatnumerot).intersection(set(lotto_rivi))
        oikein_lkm = len(oikein_menneet)
        monta_kertaa += 1
        if oikein_lkm == 4:
             nelja_oikein += 1
        elif oikein_lkm == 5:
            viisi_oikein += 1
        elif oikein_lkm == 6:
            kuusi_oikein += 1
        elif oikein_lkm == 7:
            seitseman_oikein += 1
    else:
        print("not 7 lenght")

print(f"{nelja_oikein} kertaa neljä oikein!")
print(f"{viisi_oikein} kertaa viisi oikein!")
print(f"{kuusi_oikein} kertaa kuusi oikein!")
print(f"{seitseman_oikein} kertaa seitsemän oikein!")     
print(f"arvottiin {monta_kertaa} kertaa")   
    
