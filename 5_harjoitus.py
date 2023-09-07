bitit = input("Anna bitit: ")
pyöristy = input("Anna pyöritysten lukumäärä: ")
suunta = input("Anna pyörityksen suunta (v/o): ")

try:
    pyöristy = int(pyöristy)
except:
    print("sen piti olla numero")
    
if suunta == "v":
    for i in range(pyöristy):
        bitit = bitit[1:]+bitit[0]
    print(bitit)
elif suunta == "o":
        for i in range(pyöristy):
            bitit = bitit[len(str(bitit))-1]+bitit[0:len(bitit)-1]
        print(bitit)
else:
    print("laita o tai v")
