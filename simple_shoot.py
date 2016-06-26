#!/usr/bin/env python

from __future__ import print_function

import os
from vizdoom import *
import cv2
import numpy as np

game = DoomGame()

game.load_config("../../examples/config/basic.cfg")
game.set_episode_timeout(100)

game.set_screen_format(ScreenFormat.CBCGCRDB) # BGR + Depth 4channles
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_render_hud(False)
game.set_mode(Mode.PLAYER)

actions = [[True, False, False], [False, True, False], [False, False, True]]

## action logic
def get_rect(img, depth):
    mask=np.array((img[:,:,3]==depth)).astype(np.uint8)
    img_erosion = cv2.erode(mask, np.ones((11,11),dtype=np.uint8), iterations=1)
    ret,thresh = cv2.threshold(img_erosion,0,1,cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, 1, 2)
    cnt =contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    return [x,y,w,h]

def get_centerx(img,depth):
    [x,y,w,h] = get_rect(img,depth)
    return (x+w/2)

def get_dx(img,depth0,depth1):
    return get_centerx(img,depth0)-get_centerx(img,depth1)

def get_action(img):
    dx=get_dx(img,51,0)
    if abs(dx)<20:
         return actions[2] #shoot
    if dx>0:
        return actions[1] #right
    return actions[0] #left


episodes = 10

game.init()

fourcc = cv2.VideoWriter_fourcc(*'avc1') # apple format h264
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

for i in range(episodes):
    game.new_episode()

    while not game.is_episode_finished():
        s = game.get_state()
        img = s.image_buffer
        img=(img.astype(np.uint8)).transpose(1,2,0)

        [x,y,w,h]=get_rect(img,51)
        im2=img[:,:,0:3].copy()
        imb=cv2.rectangle(im2[:,:,0:3],(x,y),(x+w,y+h),(0,255,0),2)

        [x,y,w,h]=get_rect(img,0)
        imb=cv2.rectangle(imb,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.imshow('Doom Buffer', imb)
        cv2.waitKey(2)
        out.write(imb)

        r = game.make_action(get_action(img))

    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************\n")

game.close()
out.release()


