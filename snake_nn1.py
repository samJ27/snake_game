import pygame
import random
import time
import numpy as np
import math
import tflearn
from tqdm import tqdm
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

# initialise snake parameters
dply_wdth = 500
dply_hght = 500
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 255)
wndw_clr = (255, 255, 255)

# snk_hd = [250, 250]
# snk_pos = [[250, 250], [240, 250], [230, 250]]
# apple_pos = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
score = 0
apple_img = pygame.image.load('apple.jpg')

# deploy the snake


def dply_snk(snk_pos):
    for position in snk_pos:
        pygame.draw.rect(dply, green,
                         pygame.Rect(position[0],
                                     position[1], 10, 10))

# deploy the apple


def dply_apple(apple_pos, apple_img):
    dply.blit(apple_img, (apple_pos[0], apple_pos[1]))


def starting_positions():
    snk_hd = [100, 100]
    snk_pos = [snk_hd, [90, 100], [80, 100]]
    apple_pos = [random.randrange(1, 50)*10, random.randrange(1, 50)*10]
    score = 3

    return snk_hd, snk_pos, apple_pos, score

# upon collision with apple


def cllson_wth_apple(apple_pos, score):
    apple_pos = [random.randrange(1, 20)*10, random.randrange(1, 20)*10]
    score += 1
    return apple_pos, score

# upon collision with game window edge


def cllson_wth_edges(snk_hd):
    if snk_hd[0] >= 500 or snk_hd[0] \
            <= 0 or snk_hd[1] >= 500 or snk_hd[1] <= 0:
        return 1
    else:
        return 0

# upon snake collision with it


def cllson_wth_snk(snk_pos):
    snk_hd = snk_pos[0]
    if snk_hd in snk_pos[1:]:
        return 1
    else:
        return 0


def blckd_drctns(snk_pos):
    crrnt_drctn_vctr = np.array(snk_pos[0]) - np.array(snk_pos[1])
    lft_drctn_vctr = np.array([crrnt_drctn_vctr[1], -crrnt_drctn_vctr[0]])
    rght_drctn_vctr = np.array([-crrnt_drctn_vctr[1], crrnt_drctn_vctr[0]])

    frnt_blckd = drctn_blckd(snk_pos, crrnt_drctn_vctr)
    lft_blckd = drctn_blckd(snk_pos, lft_drctn_vctr)
    rght_blckd = drctn_blckd(snk_pos, rght_drctn_vctr)

    return frnt_blckd, lft_blckd, rght_blckd

# game over if collision detected


def drctn_blckd(snk_pos, crrnt_drctn_vctr):
    # next_step = snk_pos[0] + crrnt_drctn_vctr
    snk_hd = snk_pos[0]
    if cllson_wth_edges(snk_hd) == 1 or \
            cllson_wth_snk(snk_pos) == 1:
        return 1
    else:
        return 0

# distance of the apple from the snake


def apple_dstnc_snk(apple_pos, snk_pos):
    return np.linalg.norm(np.array(apple_pos) - np.array(snk_pos[0]))

# generate the snake, detect score with apple collisions


def gnrt_snk(snk_hd, snk_pos, apple_pos, bttn_dir, score):
    if bttn_dir == 1:
        snk_hd[0] += 10
    elif bttn_dir == 0:
        snk_hd[0] -= 10
    elif bttn_dir == 2:
        snk_hd[1] += 10
    else:
        snk_hd[1] -= 10

    if snk_hd == apple_pos:
        apple_pos, score = cllson_wth_apple(apple_pos, score)
        snk_pos.insert(0, list(snk_hd))

    else:
        snk_pos.insert(0, list(snk_hd))
        snk_pos.pop()

    return snk_pos, apple_pos, score

# generate next direction for snake


def nxt_direction(snk_pos, angle):
    direction = 0
    if angle > 0:
        direction = 1
    elif angle < 0:
        direction = -1
    else:
        direction = 0

    # print(.direction)

    crrnt_drctn_vctr = np.array(snk_pos[0]) - np.array(snk_pos[1])
    lft_drctn_vctr = np.array([crrnt_drctn_vctr[1], -crrnt_drctn_vctr[0]])
    rght_drctn_vctr = np.array([-crrnt_drctn_vctr[1], crrnt_drctn_vctr[0]])

    nxt_drctn = crrnt_drctn_vctr
    # print(.nxt_drctn)
    if direction == -1:
        nxt_drctn = lft_drctn_vctr
    if direction == 1:
        nxt_drctn = rght_drctn_vctr

    bttn_dir = gnrt_bttn_drctn(nxt_drctn)

    return direction, bttn_dir

# determine button press for game


def gnrt_bttn_drctn(nxt_drctn):
    bttn_dir = 0
    if nxt_drctn.tolist() == [10, 0]:
        bttn_dir = 1
    elif nxt_drctn.tolist() == [-10, 0]:
        bttn_dir = 0
    elif nxt_drctn.tolist() == [0, 10]:
        bttn_dir = 2
    else:
        bttn_dir = 3

    return bttn_dir

# generate angle with apple for training


def apple_ang(snk_pos, apple_pos):
    apple_drctn = np.array(apple_pos)-np.array(snk_pos[0])
    snk_drctn = np.array(snk_pos[0])-np.array(snk_pos[1])

    # normalise the snake and apple direction vectors
    norm_apple_drctn = np.linalg.norm(apple_drctn)
    norm_snk_drctn = np.linalg.norm(snk_drctn)
    if norm_apple_drctn == 0:
        norm_apple_drctn = 10
    if norm_snk_drctn == 0:
        norm_snk_drctn = 10

    nrmlsd_apple = apple_drctn / norm_apple_drctn
    nrmlsd_snk = snk_drctn / norm_snk_drctn
    angle = math.atan2(nrmlsd_apple[1] * nrmlsd_snk[0] -
                       nrmlsd_apple[0] * nrmlsd_snk[1],
                       nrmlsd_apple[1] * nrmlsd_snk[1] +
                       nrmlsd_apple[0] * nrmlsd_snk[0]) / math.pi
    # print(.angle)
    return angle

# main game loop


def game_loop(snk_pos, snk_hd, apple, apple_pos, bttn_dir, score):
    crash = False
    while crash is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crash = True
        dply.fill(wndw_clr)
        dply_apple(apple_pos, apple)
        dply_snk(snk_pos)

        snk_pos, apple_pos, score = gnrt_snk(snk_hd,
                                             snk_pos,
                                             apple_pos,
                                             bttn_dir,
                                             score)
        pygame.display.set_caption("Snake" + "  " + "Score:  " + str(score))
        pygame.display.update()
        # .prev_bttn_dir = .bttn_dir
        """if .drctn_blckd(.snk_pos, .crrnt_drctn_vctr) == 1:
            .crash = True"""

        clock = pygame.time.Clock()
        clock.tick(20)
        return snk_pos, apple_pos, score


def trngn_data():
    trng_data_x = []
    trng_data_y = []
    trng_games = 20
    steps_per_game = 500

    for _ in tqdm(range(trng_games)):
        snk_hd, snk_pos, apple_pos, score = starting_positions()
        prv_apple_dtnc = apple_dstnc_snk(apple_pos, snk_pos)
        prev_score = score

        for _ in range(steps_per_game):
            angle = apple_ang(snk_pos, apple_pos)
            direction, bttn_dir = nxt_direction(snk_pos, angle)
            snk_pos, apple_pos, score = game_loop(snk_hd,
                                                  snk_pos, apple_pos,
                                                  bttn_dir, score, apple_img)
            frnt_blckd, lft_blckd, rght_blckd = blckd_drctns(snk_pos)
            trng_data_x.append([lft_blckd, frnt_blckd,
                                rght_blckd, angle, direction])
            if cllson_wth_edges(snk_pos[0]) == 1 or cllson_wth_snk(snk_pos) == 1:
                trng_data_y.append(-1)
                break
            else:
                crrnt_snk_dstnc = apple_dstnc_snk(apple_pos, snk_pos)
                if score > prev_score or crrnt_snk_dstnc < prv_apple_dtnc:
                    trng_data_y.append(1)
                else:
                    trng_data_y.append(0)
                prv_apple_dtnc = crrnt_snk_dstnc
                prev_score = score

        return trng_data_x, trng_data_y


def game_wth_NN(model):
    max_scr = 3
    avg_scr = 0
    test_games = 300
    test_steps = 300

    for _ in tqdm(range(test_games)):
        snk_hd, snk_pos, apple_pos, score = starting_positions()

        for _ in range(test_steps):
            frnt_blckd, lft_blckd, rght_blckd = blckd_drctns(
                snk_pos)
            angle = apple_ang(snk_pos, apple_pos)
            predictions = []
            for i in range(-1, 2):
                predictions.append(model.predict(
                    np.array([lft_blckd, frnt_blckd, rght_blckd, angle, i]).reshape(-1, 5, 1)))
            predicted_drctn = np.argmax(np.array(predictions)) - 1

            nxt_drctn = np.array(snk_pos[0]) - np.array(snk_pos[1])
            if predicted_drctn == -1:
                nxt_drctn = np.array(
                    [nxt_drctn[1], -nxt_drctn[0]])
            if predicted_drctn == 1:
                nxt_drctn = np.array(
                    [-nxt_drctn[1], nxt_drctn[0]])

            bttn_dir = gnrt_bttn_drctn(nxt_drctn)

            snk_pos, apple_pos, score = game_loop(
                snk_hd, snk_pos, apple_pos, bttn_dir, score, apple_img)
            if cllson_wth_edges(snk_pos[0]) == 1 or cllson_wth_snk(snk_pos) == 1:
                avg_scr += score
                break

            if score > max_scr:
                max_scr = score

    return max_scr, avg_scr/300


def train_model():
    neural_network = input_data(shape=[None, 5, 1], name='input')
    neural_network = fully_connected(neural_network, 25, activation='relu')
    neural_network = fully_connected(neural_network, 10, activation='relu')
    neural_network = fully_connected(neural_network, 1, activation='tanh')
    neural_network = regression(
        neural_network, optimizer='adam', learning_rate=1e-3, loss='mean_square', name='target')
    model = tflearn.DNN(neural_network)

    return model

    # initialise game
    # def on_init():
pygame.init()
dply = pygame.display.set_mode((dply_wdth, dply_hght))
dply.fill(wndw_clr)
pygame.display.update

'''.fnl_score = .game_loop(.snk_hd, .snk_pos, .apple_pos,
                                1, .apple_img, .score)'''
dply_txt = "Score: " + str(score)

trng_data_x, trng_data_y = trngn_data()
model = train_model()
# print(.trng_data_x)
model.fit(np.array(trng_data_x).reshape(-1, 5, 1),
          np.array(trng_data_y).reshape(-1, 1), n_epoch=3, shuffle=True)
max_scr, avg_scr = game_wth_NN(model)


'''if __name__ == "__main__":
    app = snakeApp()
    app.on_init()'''
