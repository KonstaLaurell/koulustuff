import pygame

def kontrolli(hahmot, tapahtuma, nopeus):

    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_RIGHT]:
        päähahmo = hahmot[0]
        päähahmo[1] += nopeus 
    if pressed[pygame.K_DOWN]:
        päähahmo = hahmot[0]
        päähahmo[2] += nopeus
    if pressed[pygame.K_LEFT]:
        päähahmo = hahmot[0]
        päähahmo[1] -= nopeus    
    if pressed[pygame.K_UP]:
        päähahmo = hahmot[0]
        päähahmo[2] -= nopeus
