import pygame

def kontrolli(hahmot, tapahtuma, nopeus):

    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_RIGHT or pygame.K_d]:
        päähahmo = hahmot[0]
        päähahmo[1] += nopeus 
    if pressed[pygame.K_DOWN or pygame.K_s]:
        päähahmo = hahmot[0]
        päähahmo[2] += nopeus
    if pressed[pygame.K_LEFT or pygame.K_a]:
        päähahmo = hahmot[0]
        päähahmo[1] -= nopeus    
    if pressed[pygame.K_UP or pygame.K_w]:
        päähahmo = hahmot[0]
        päähahmo[2] -= nopeus
