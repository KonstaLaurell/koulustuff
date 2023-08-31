luku= input("Anna luku: ")
luku = int(luku)
def OnkoAlkuluku(luku):
  for i in range(2,luku//2+1):
    if (luku % i) == 0:
      return 'ei ole alkuluku'
  return 'on alkuluku'




print(OnkoAlkuluku(luku))


