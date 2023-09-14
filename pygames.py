import pygame , os, controll
pygame.init()
info = pygame.display.Info() # You have to call this before pygame.display.set_mode()
screen_width,screen_height = info.current_w,info.current_h
window_width,window_height = screen_width-10,screen_height-50
naytto = pygame.display.set_mode((window_width,window_height))
pygame.display.set_caption("Piirtäminen")
nopeus = 5
def piirraKuva(kuvatiedosto, x, y):
    naytto.blit(kuvatiedosto, (x, y))
def piirtaminen(naytto, hahmot):
    naytto.fill((10, 100, 255))
    for hahmo in hahmot:
        if hahmo[3] == True:
            kuva = pygame.image.load(hahmo[0]).convert()
            kuva.set_colorkey((0,0,0))
            naytto.blit(kuva, (hahmo[1], hahmo[2]))
    pygame.display.flip()

def main():
    kissahahmo = ["smurf.png", 10, 20, True]
    hahmot = [kissahahmo]

    while True:

        tapahtuma = pygame.event.poll()
        if tapahtuma.type == pygame.QUIT:
            break
        controll.kontrolli(hahmot, tapahtuma, nopeus)
        piirtaminen(naytto, hahmot)

main() 