import pygame , os, controll, random
pygame.init()
info = pygame.display.Info() # You have to call this before pygame.display.set_mode()
screen_width,screen_height = info.current_w,info.current_h
window_width,window_height = screen_width-10,screen_height-50
naytto = pygame.display.set_mode((window_width,window_height))
pygame.display.set_caption("game")
nopeus = 1
def piirraKuva(kuvatiedosto, x, y):
    naytto.blit(kuvatiedosto, (x, y))
def piirtaminen(naytto, hahmot):
    naytto.fill((55, 66, 91))
    for hahmo in hahmot:
        if hahmo[4] == True:
            kuva = pygame.image.load(hahmo[0]).convert()
            kuva.set_colorkey((0,0,0))
            kuva = pygame.transform.flip(kuva,True,False)
            kuva = pygame.transform.scale(kuva, (244/1.5,407/1.5))
            naytto.blit(kuva, (hahmo[1], hahmo[2]))
        elif hahmo[4] == False:
            kuva = pygame.image.load(hahmo[0]).convert()
            kuva.set_colorkey((0,0,0))
            kuva = pygame.transform.flip(kuva,False,False)
            kuva = pygame.transform.scale(kuva, (244/1.5,407/1.5))
            naytto.blit(kuva, (hahmo[1], hahmo[2]))
    pygame.display.flip()

def kontrolli(hahmot, tapahtuma, nopeus ):

    pressed = pygame.key.get_pressed()

    if pressed[pygame.K_DOWN]:
        päähahmo = hahmot[0]
        päähahmo[2] += nopeus  
    if pressed[pygame.K_UP]:
        päähahmo = hahmot[0]
        päähahmo[2] -= nopeus

def main():
    kissahahmo = ["smurf.png", 10, 10, True,True]
    hahmot = [kissahahmo]

    while True:

        tapahtuma = pygame.event.poll()
        if tapahtuma.type == pygame.QUIT:
            break
        kontrolli(hahmot, tapahtuma, nopeus)
        piirtaminen(naytto, hahmot)
main() 